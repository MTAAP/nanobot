import fs from 'fs';
import path from 'path';

export enum AgentState {
    IDLE = 'IDLE',
    INIT = 'INIT',
    WORKING = 'WORKING',
    VERIFYING = 'VERIFYING',
    COMPLETED = 'COMPLETED',
    FAILED = 'FAILED'
}

export interface AgentStatus {
    agentId: string;
    taskId: string;
    state: AgentState;
    lastPulse: number;
    proofOfWork?: string;
    error?: string;
}

export class AgentRegistry {
    private registryDir: string;

    constructor(workspaceDir: string) {
        this.registryDir = path.join(workspaceDir, '.agents');
        if (!fs.existsSync(this.registryDir)) {
            fs.mkdirSync(this.registryDir, { recursive: true });
        }
    }

    public register(status: AgentStatus): void {
        const filePath = path.join(this.registryDir, `${status.agentId}.json`);
        fs.writeFileSync(filePath, JSON.stringify(status, null, 2));
    }

    public getStatus(agentId: string): AgentStatus | null {
        const filePath = path.join(this.registryDir, `${agentId}.json`);
        if (!fs.existsSync(filePath)) return null;
        return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    }

    public updatePulse(agentId: string): void {
        const status = this.getStatus(agentId);
        if (status) {
            status.lastPulse = Date.now();
            this.register(status);
        }
    }
}
