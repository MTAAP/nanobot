/* Chat sidebar WebSocket client */

let chatWs = null;
let chatSessionId = null;
let reconnectTimer = null;
let streamingMsgEl = null;
let inActiveChat = false;

function generateSessionId() {
    return 'web-' + Date.now().toString(36)
        + '-' + Math.random().toString(36).slice(2, 8);
}

function renderMarkdown(text) {
    text = (text || '')
        .replace(/<\|tool_calls_section_begin\|>[\s\S]*/g, '')
        .trimEnd();
    if (!text) return '';
    if (typeof marked !== 'undefined') {
        return marked.parse(text);
    }
    // Fallback: escape HTML and preserve whitespace
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/\n/g, '<br>');
}

function openChat(context, id) {
    const sidebar = document.getElementById('chat-sidebar');
    if (!sidebar) return;

    sidebar.classList.remove('hidden');
    document.body.classList.add('chat-open');

    if (context && id) {
        // Context-specific chat: create new session immediately
        inActiveChat = true;
        const titleEl = document.getElementById('chat-title');
        const labels = {
            signal: 'Signal',
            lead: 'Lead',
            recommendation: 'Recommendation'
        };
        titleEl.textContent =
            'Discuss ' + (labels[context] || context) + ' #' + id;

        const messagesEl = document.getElementById('chat-messages');
        messagesEl.innerHTML = '';
        streamingMsgEl = null;

        showInputArea();
        chatSessionId = generateSessionId();
        connectWebSocket(context, id);
    } else {
        // No context: show session list
        showSessionList();
    }
}

function closeSidebar() {
    inActiveChat = false;
    const sidebar = document.getElementById('chat-sidebar');
    if (sidebar) {
        sidebar.classList.add('hidden');
        document.body.classList.remove('chat-open');
    }
    disconnectWebSocket();
    hideBackButton();
}

function showSessionList() {
    inActiveChat = false;
    const titleEl = document.getElementById('chat-title');
    titleEl.textContent = 'Chat with Nano';
    hideBackButton();
    hideInputArea();

    const messagesEl = document.getElementById('chat-messages');
    messagesEl.innerHTML = '<div class="chat-status">Loading...</div>';

    fetch('/chat/sessions')
        .then(function(r) { return r.json(); })
        .then(function(sessions) {
            messagesEl.innerHTML = '';

            var newBtn = document.createElement('button');
            newBtn.className = 'chat-new-btn';
            newBtn.textContent = 'New Chat';
            newBtn.onclick = startNewChat;
            messagesEl.appendChild(newBtn);

            if (!sessions || sessions.length === 0) {
                var empty = document.createElement('div');
                empty.className = 'chat-status';
                empty.textContent = 'No previous chats';
                messagesEl.appendChild(empty);
                return;
            }

            sessions.forEach(function(s) {
                var item = document.createElement('div');
                item.className = 'chat-session-item';
                item.onclick = function() {
                    openSession(s.id);
                };

                var info = document.createElement('div');
                info.className = 'chat-session-info';

                var preview = document.createElement('div');
                preview.className = 'chat-session-preview';
                preview.textContent = s.preview || 'Chat session';

                var meta = document.createElement('div');
                var time = document.createElement('span');
                time.className = 'chat-session-time';
                time.textContent = formatTimeAgo(s.updated_at);
                var count = document.createElement('span');
                count.className = 'chat-session-count';
                count.textContent =
                    ' - ' + (s.message_count || 0) + ' msgs';

                meta.appendChild(time);
                meta.appendChild(count);
                info.appendChild(preview);
                info.appendChild(meta);

                var delBtn = document.createElement('button');
                delBtn.className = 'chat-session-delete';
                delBtn.innerHTML = '&times;';
                delBtn.onclick = function(e) {
                    e.stopPropagation();
                    deleteSession(s.id, item);
                };

                item.appendChild(info);
                item.appendChild(delBtn);
                messagesEl.appendChild(item);
            });
        })
        .catch(function() {
            messagesEl.innerHTML = '';
            var err = document.createElement('div');
            err.className = 'chat-status';
            err.textContent = 'Failed to load sessions';
            messagesEl.appendChild(err);
        });
}

function startNewChat() {
    inActiveChat = true;
    chatSessionId = generateSessionId();
    var titleEl = document.getElementById('chat-title');
    titleEl.textContent = 'Chat with Nano';

    var messagesEl = document.getElementById('chat-messages');
    messagesEl.innerHTML = '';
    streamingMsgEl = null;

    showBackButton();
    showInputArea();
    connectWebSocket();
}

function openChatAndSend(prompt) {
    var sidebar = document.getElementById('chat-sidebar');
    if (!sidebar) return;
    sidebar.classList.remove('hidden');
    document.body.classList.add('chat-open');
    startNewChat();
    setTimeout(function() {
        var input = document.getElementById('chat-input');
        input.value = prompt;
        sendMessage();
    }, 500);
}

function openSession(sessionId) {
    inActiveChat = true;
    chatSessionId = sessionId;
    var titleEl = document.getElementById('chat-title');
    titleEl.textContent = 'Chat with Nano';

    var messagesEl = document.getElementById('chat-messages');
    messagesEl.innerHTML = '';
    streamingMsgEl = null;

    showBackButton();
    showInputArea();

    fetch('/chat/sessions/' + sessionId + '/history')
        .then(function(r) { return r.json(); })
        .then(function(data) {
            (data.messages || []).forEach(function(msg) {
                appendMessage(
                    msg.role === 'user' ? 'user' : 'nano',
                    msg.content
                );
            });
            connectWebSocket();
        })
        .catch(function() {
            appendStatus('Failed to load history');
            connectWebSocket();
        });
}

function showBackButton() {
    hideBackButton();
    var header = document.querySelector('.chat-header');
    if (!header) return;
    var btn = document.createElement('button');
    btn.className = 'chat-back-btn';
    btn.innerHTML = '&#8592;';
    btn.onclick = goBackToSessions;
    header.insertBefore(btn, header.firstChild);
}

function hideBackButton() {
    var btn = document.querySelector('.chat-back-btn');
    if (btn) btn.remove();
}

function goBackToSessions() {
    disconnectWebSocket();
    hideBackButton();
    showSessionList();
}

function deleteSession(sessionId, itemEl) {
    fetch('/chat/sessions/' + sessionId, {method: 'DELETE'})
        .then(function() {
            if (itemEl) itemEl.remove();
        })
        .catch(function() {
            // Silently ignore delete errors
        });
}

function showInputArea() {
    var area = document.querySelector('.chat-input-area');
    if (area) area.style.display = 'flex';
}

function hideInputArea() {
    var area = document.querySelector('.chat-input-area');
    if (area) area.style.display = 'none';
}

function formatTimeAgo(isoDate) {
    if (!isoDate) return '';
    var date = new Date(isoDate);
    var now = new Date();
    var diffMs = now - date;
    var diffSec = Math.floor(diffMs / 1000);
    var diffMin = Math.floor(diffSec / 60);
    var diffHr = Math.floor(diffMin / 60);
    var diffDay = Math.floor(diffHr / 24);

    if (diffSec < 60) return 'just now';
    if (diffMin < 60) return diffMin + 'm ago';
    if (diffHr < 24) return diffHr + 'h ago';
    if (diffDay < 7) return diffDay + 'd ago';

    var months = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ];
    return months[date.getMonth()] + ' ' + date.getDate();
}

function connectWebSocket(context, id) {
    disconnectWebSocket();

    var protocol =
        window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    var url = protocol + '//' + window.location.host
        + '/chat/ws/' + chatSessionId;

    if (context && id) {
        url += '?context=' + encodeURIComponent(context)
            + '&id=' + encodeURIComponent(id);
    }

    chatWs = new WebSocket(url);

    chatWs.onopen = function() {
        clearTimeout(reconnectTimer);
        if (inActiveChat) {
            appendStatus('Connected');
        }
    };

    chatWs.onmessage = function(event) {
        try {
            var data = JSON.parse(event.data);
            if (data.type === 'message') {
                if (streamingMsgEl && data.content) {
                    streamingMsgEl
                        .querySelector('.chat-msg-content')
                        .innerHTML = renderMarkdown(data.content);
                    streamingMsgEl = null;
                } else if (streamingMsgEl && !data.content) {
                    streamingMsgEl = null;
                } else if (data.content) {
                    appendMessage('nano', data.content);
                }
                hideTyping();
            } else if (data.type === 'progress') {
                if (data.kind === 'streaming' && data.detail) {
                    if (!streamingMsgEl) {
                        streamingMsgEl =
                            createMessageEl('nano', data.detail);
                    } else {
                        streamingMsgEl
                            .querySelector('.chat-msg-content')
                            .innerHTML =
                                renderMarkdown(data.detail);
                    }
                    scrollToBottom();
                } else {
                    showTyping(
                        data.detail
                            || data.tool_name
                            || 'Thinking...'
                    );
                }
            } else if (data.type === 'error') {
                appendStatus('Error: ' + data.content);
            }
        } catch (e) {
            appendMessage('nano', event.data);
        }
    };

    chatWs.onclose = function() {
        if (inActiveChat) {
            appendStatus('Disconnected');
            reconnectTimer = setTimeout(function() {
                if (inActiveChat) {
                    connectWebSocket(context, id);
                }
            }, 3000);
        }
    };

    chatWs.onerror = function() {
        appendStatus('Connection error');
    };
}

function disconnectWebSocket() {
    inActiveChat = false;
    clearTimeout(reconnectTimer);
    if (chatWs) {
        chatWs.close();
        chatWs = null;
    }
    streamingMsgEl = null;
}

function sendMessage() {
    var input = document.getElementById('chat-input');
    var content = input.value.trim();
    if (!content || !chatWs
        || chatWs.readyState !== WebSocket.OPEN) return;

    appendMessage('user', content);
    chatWs.send(JSON.stringify({content: content}));
    input.value = '';
    showTyping('Thinking...');
}

function createMessageEl(role, content) {
    var messagesEl = document.getElementById('chat-messages');
    var msgDiv = document.createElement('div');
    msgDiv.className = 'chat-msg chat-msg-' + role;

    var labelEl = document.createElement('div');
    labelEl.className = 'chat-msg-label';
    labelEl.textContent = role === 'user' ? 'You' : 'Nano';

    var contentEl = document.createElement('div');
    contentEl.className = 'chat-msg-content';

    if (role === 'user') {
        contentEl.textContent = content;
    } else {
        contentEl.innerHTML = renderMarkdown(content);
    }

    msgDiv.appendChild(labelEl);
    msgDiv.appendChild(contentEl);
    messagesEl.appendChild(msgDiv);
    scrollToBottom();
    return msgDiv;
}

function appendMessage(role, content) {
    createMessageEl(role, content);
}

function appendStatus(text) {
    var messagesEl = document.getElementById('chat-messages');
    var statusEl = document.createElement('div');
    statusEl.className = 'chat-status';
    statusEl.textContent = text;
    messagesEl.appendChild(statusEl);
    scrollToBottom();
}

function scrollToBottom() {
    var messagesEl = document.getElementById('chat-messages');
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function showTyping(detail) {
    var typingEl = document.getElementById('chat-typing');
    if (!typingEl) {
        typingEl = document.createElement('div');
        typingEl.id = 'chat-typing';
        typingEl.className = 'chat-typing';
        document.getElementById('chat-messages')
            .appendChild(typingEl);
    }
    typingEl.textContent = detail;
    typingEl.style.display = 'block';
    scrollToBottom();
}

function hideTyping() {
    var typingEl = document.getElementById('chat-typing');
    if (typingEl) {
        typingEl.style.display = 'none';
    }
}

/* Intelligence scan launchers */

function launchScan() {
    var topics =
        document.getElementById('scan-topics')?.value || '';
    var region =
        document.getElementById('scan-region')?.value || '';
    var industry =
        document.getElementById('scan-industry')?.value || '';
    var service =
        document.getElementById('scan-service')?.value || '';

    var prompt = 'Scan for new market intelligence signals';
    var refinements = [];
    if (topics) refinements.push('focused on: ' + topics);
    if (region) refinements.push('in ' + region);
    if (industry) {
        refinements.push(
            'in the ' + industry + ' industry'
        );
    }
    if (service) {
        refinements.push(
            'relevant to our ' + service + ' practice'
        );
    }

    if (refinements.length) {
        prompt += ' ' + refinements.join(', ');
    }
    prompt += '. Use market_intelligence deep_scan to perform'
        + ' a comprehensive search, fetch articles,'
        + ' and extract signals. Report what you find.';

    openChatAndSend(prompt);
}

function launchCompanyScan() {
    var company =
        document.getElementById('company-name')?.value || '';
    if (!company) return;
    openChatAndSend(
        'Deep dive on ' + company
        + '. Use scan_company to find recent news,'
        + ' then analyze_signals to extract'
        + ' structured signals.'
        + ' Score the lead and suggest K&P services.'
    );
}

function launchScoring() {
    openChatAndSend(
        'Score all current leads and match'
        + ' top leads to consultants.'
        + ' Generate recommendations for hot leads.'
        + ' Use score_all and match_consultant tools.'
    );
}

/* Compose page helpers */

function openInEmailClient() {
    var recipient =
        document.getElementById('recipient')?.value || '';
    var subject =
        document.getElementById('subject')?.value || '';
    var body =
        document.getElementById('email-body')?.value || '';
    var mailto = 'mailto:' + encodeURIComponent(recipient)
        + '?subject=' + encodeURIComponent(subject)
        + '&body=' + encodeURIComponent(body);
    window.location.href = mailto;
}

function regenerateEmail() {
    var company =
        document.getElementById('compose-company')
            ?.value || '';
    var anrede =
        document.getElementById('anrede')?.value || '';
    var nachname =
        document.getElementById('nachname')?.value || '';
    openChatAndSend(
        'Regenerate the outreach email for ' + company
        + '. Address it to ' + anrede + ' ' + nachname
        + '. Make it compelling and specific'
        + ' to their situation.'
    );
}

// Handle Enter key in textarea
document.addEventListener('DOMContentLoaded', function() {
    var input = document.getElementById('chat-input');
    if (input) {
        input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }
    // Start notification polling
    pollUnreadCount();
    setInterval(pollUnreadCount, 30000);

    // Close dropdown on outside click
    document.addEventListener('click', function(e) {
        var wrapper = document.getElementById('notif-wrapper');
        var dropdown = document.getElementById('notif-dropdown');
        if (wrapper && dropdown && !wrapper.contains(e.target)) {
            dropdown.classList.add('hidden');
        }
    });
});

/* Notification bell */

function pollUnreadCount() {
    fetch('/notifications/api/unread-count')
        .then(function(r) { return r.json(); })
        .then(function(data) {
            var badge = document.getElementById('notif-badge');
            if (!badge) return;
            var count = data.count || 0;
            if (count > 0) {
                badge.textContent = count > 99 ? '99+' : count;
                badge.classList.remove('hidden');
            } else {
                badge.classList.add('hidden');
            }
        })
        .catch(function() {});
}

function toggleNotifications() {
    var dropdown = document.getElementById('notif-dropdown');
    if (!dropdown) return;

    if (dropdown.classList.contains('hidden')) {
        dropdown.classList.remove('hidden');
        loadNotifications();
        // Mark all as read
        fetch('/notifications/api/mark-read', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({})
        }).then(function() {
            var badge = document.getElementById('notif-badge');
            if (badge) badge.classList.add('hidden');
        }).catch(function() {});
    } else {
        dropdown.classList.add('hidden');
    }
}

function loadNotifications() {
    var dropdown = document.getElementById('notif-dropdown');
    if (!dropdown) return;

    dropdown.innerHTML =
        '<div class="px-3 py-2 text-xs text-kp-text">Loading...</div>';

    fetch('/notifications/api/recent')
        .then(function(r) { return r.json(); })
        .then(function(items) {
            if (!items || items.length === 0) {
                dropdown.innerHTML =
                    '<div class="px-3 py-4 text-sm text-kp-text'
                    + ' text-center">No notifications</div>';
                return;
            }
            dropdown.innerHTML = '';
            items.forEach(function(n) {
                var el = document.createElement('div');
                el.className = 'notif-item'
                    + (n.read ? '' : ' notif-unread');

                var catBadge = '';
                if (n.category === 'cron_ok') {
                    catBadge = '<span class="badge-ok">ok</span>';
                } else if (n.category === 'cron_error') {
                    catBadge =
                        '<span class="badge-error">error</span>';
                } else if (n.category === 'scan_complete') {
                    catBadge = '<span class="badge-ok">scan</span>';
                }

                var bodyHtml = n.body
                    ? '<div class="text-xs text-kp-text truncate">'
                      + escapeHtml(n.body) + '</div>'
                    : '';

                el.innerHTML = catBadge
                    + '<div class="flex-1 min-w-0">'
                    + '<div class="text-sm text-kp-dark truncate">'
                    + escapeHtml(n.title) + '</div>'
                    + bodyHtml
                    + '<div class="text-xs text-kp-text mt-0.5">'
                    + escapeHtml(n.created_at || '') + '</div>'
                    + '</div>';

                if (n.link) {
                    el.onclick = function() {
                        window.location.href = n.link;
                    };
                }
                dropdown.appendChild(el);
            });
        })
        .catch(function() {
            dropdown.innerHTML =
                '<div class="px-3 py-4 text-sm text-kp-text'
                + ' text-center">Failed to load</div>';
        });
}

function escapeHtml(text) {
    var el = document.createElement('span');
    el.textContent = text;
    return el.innerHTML;
}
