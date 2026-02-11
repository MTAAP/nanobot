Sehr geehrte/r {{ anrede|default('Frau/Herr') }} {{ nachname|default('') }},

die Optimierung von Geschäftsprozessen ist ein zentraler Hebel für nachhaltige Wettbewerbsfähigkeit{% if signal_description %} -- ein Thema, das offenbar auch für {{ company_name }} relevant ist: {{ signal_description }}{% endif %}.

Bei Kraus & Partner verbinden wir Prozessexzellenz mit Veränderungskompetenz. Denn die besten Prozesse nutzen wenig, wenn die Organisation sie nicht lebt. Wir helfen Ihnen, beides zu erreichen.

{% if consultant_name %}{{ consultant_name }} hat zahlreiche Prozessoptimierungsprojekte{% if industry %} in der {{ industry }}{% endif %} erfolgreich begleitet und steht für ein Gespräch bereit.{% endif %}

Dürfen wir einen Termin vereinbaren?

Mit freundlichen Grüßen
{{ absender_name }}
Kraus & Partner

---
{{ impressum }}

Sie möchten keine weiteren Nachrichten erhalten? {{ abmeldelink }}
