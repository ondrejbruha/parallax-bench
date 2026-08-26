"""Generate smoke queries.jsonl + qrels.txt from the hand-authored table below.

Smoke queries are hand-authored (English pivot, hand-translated to cs/de) —
no LLM involved; see benchmark/smoke/generation.md.  Each query is written so
its answer is unambiguously in exactly one of the ten smoke documents.

Usage: python benchmark/build/make_smoke_queries.py
"""

from __future__ import annotations

import json
from pathlib import Path

SMOKE = Path(__file__).resolve().parent.parent / "smoke"

# (query_group, source_celex, {lang: text})
QUERIES: list[tuple[str, str, dict[str, str]]] = [
    ("q00001", "31985L0374", {
        "en": "After how many years from the product being put into circulation do the injured person's rights against the producer of a defective product expire?",
        "cs": "Po kolika letech od uvedení výrobku do oběhu zanikají práva poškozeného vůči výrobci vadného výrobku?",
        "de": "Nach wie vielen Jahren ab dem Inverkehrbringen des Produkts erlöschen die Ansprüche des Geschädigten gegen den Hersteller eines fehlerhaften Produkts?",
    }),
    ("q00002", "31985L0374", {
        "en": "What lower threshold of damage to property applies before a producer is liable for damage caused by a defective product?",
        "cs": "Jaká spodní hranice škody na majetku platí pro odpovědnost výrobce za škodu způsobenou vadným výrobkem?",
        "de": "Welche untere Schwelle für Sachschäden gilt, bevor ein Hersteller für durch ein fehlerhaftes Produkt verursachte Schäden haftet?",
    }),
    ("q00003", "31993L0013", {
        "en": "What happens to the rest of a consumer contract when one of its terms is found unfair and not binding on the consumer?",
        "cs": "Co se stane se zbytkem spotřebitelské smlouvy, když je jedna z jejích podmínek shledána zneužívající a pro spotřebitele nezávaznou?",
        "de": "Was geschieht mit dem übrigen Verbrauchervertrag, wenn eine seiner Klauseln als missbräuchlich und für den Verbraucher unverbindlich eingestuft wird?",
    }),
    ("q00004", "31993L0013", {
        "en": "Which interpretation of a written contractual term prevails when there is doubt about its meaning in a consumer contract?",
        "cs": "Který výklad písemné smluvní podmínky má přednost, existuje-li pochybnost o jejím významu ve spotřebitelské smlouvě?",
        "de": "Welche Auslegung einer schriftlichen Vertragsklausel hat Vorrang, wenn Zweifel über ihre Bedeutung in einem Verbrauchervertrag bestehen?",
    }),
    ("q00005", "32000L0031", {
        "en": "Under what conditions is a hosting service provider exempt from liability for information stored at the request of a recipient of the service?",
        "cs": "Za jakých podmínek je poskytovatel hostingové služby zproštěn odpovědnosti za informace ukládané na žádost příjemce služby?",
        "de": "Unter welchen Bedingungen ist ein Hosting-Anbieter von der Verantwortlichkeit für die im Auftrag eines Nutzers gespeicherten Informationen befreit?",
    }),
    ("q00006", "32000L0031", {
        "en": "May member states impose on internet service providers a general obligation to monitor the information which they transmit or store?",
        "cs": "Smějí členské státy uložit poskytovatelům internetových služeb obecnou povinnost dohlížet na informace, které přenášejí nebo ukládají?",
        "de": "Dürfen Mitgliedstaaten Internetdienstleistern eine allgemeine Verpflichtung auferlegen, die von ihnen übermittelten oder gespeicherten Informationen zu überwachen?",
    }),
    ("q00007", "32001L0029", {
        "en": "Which temporary acts of reproduction, such as caching, are exempted from the author's reproduction right?",
        "cs": "Které dočasné úkony rozmnožování, například ukládání do mezipaměti, jsou vyňaty z autorova práva na rozmnožování?",
        "de": "Welche vorübergehenden Vervielfältigungshandlungen, etwa das Caching, sind vom Vervielfältigungsrecht des Urhebers ausgenommen?",
    }),
    ("q00008", "32001L0029", {
        "en": "What legal protection must be provided against the circumvention of effective technological protection measures for copyrighted works?",
        "cs": "Jaká právní ochrana musí být poskytnuta proti obcházení účinných technologických prostředků ochrany autorských děl?",
        "de": "Welcher rechtliche Schutz muss gegen die Umgehung wirksamer technischer Schutzmaßnahmen für urheberrechtlich geschützte Werke gewährt werden?",
    }),
    ("q00009", "32002L0058", {
        "en": "Under what condition may a provider process traffic data for the purpose of marketing its electronic communications services?",
        "cs": "Za jaké podmínky smí poskytovatel zpracovávat provozní údaje pro účely marketingu svých služeb elektronických komunikací?",
        "de": "Unter welcher Bedingung darf ein Anbieter Verkehrsdaten zum Zweck der Vermarktung seiner elektronischen Kommunikationsdienste verarbeiten?",
    }),
    ("q00010", "32002L0058", {
        "en": "What is required before information such as cookies may be stored in the terminal equipment of a subscriber or user?",
        "cs": "Co je vyžadováno předtím, než smějí být v koncovém zařízení účastníka nebo uživatele ukládány informace, jako jsou cookies?",
        "de": "Was ist erforderlich, bevor Informationen wie Cookies im Endgerät eines Teilnehmers oder Nutzers gespeichert werden dürfen?",
    }),
    ("q00011", "32005L0029", {
        "en": "When is a commercial practice considered a misleading omission towards consumers?",
        "cs": "Kdy se obchodní praktika považuje za klamavé opomenutí vůči spotřebitelům?",
        "de": "Wann gilt eine Geschäftspraxis als irreführende Unterlassung gegenüber Verbrauchern?",
    }),
    ("q00012", "32005L0029", {
        "en": "By which three forms of pressure — such as harassment or coercion — is a commercial practice regarded as aggressive?",
        "cs": "Kterými třemi formami nátlaku — jako je obtěžování nebo donucování — se obchodní praktika považuje za agresivní?",
        "de": "Durch welche drei Formen der Druckausübung — etwa Belästigung oder Nötigung — gilt eine Geschäftspraxis als aggressiv?",
    }),
    ("q00013", "32006L0114", {
        "en": "Under what conditions is comparative advertising permitted as far as the comparison is concerned?",
        "cs": "Za jakých podmínek je srovnávací reklama dovolena, pokud jde o srovnání?",
        "de": "Unter welchen Bedingungen ist vergleichende Werbung zulässig, was den Vergleich anbelangt?",
    }),
    ("q00014", "32006L0114", {
        "en": "Whom does the directive on misleading and comparative advertising primarily protect against misleading advertising?",
        "cs": "Koho směrnice o klamavé a srovnávací reklamě především chrání před klamavou reklamou?",
        "de": "Wen schützt die Richtlinie über irreführende und vergleichende Werbung in erster Linie vor irreführender Werbung?",
    }),
    ("q00015", "32011L0083", {
        "en": "How many days does a consumer have to withdraw from a distance or off-premises contract without giving any reason?",
        "cs": "Kolik dnů má spotřebitel na odstoupení od smlouvy uzavřené na dálku nebo mimo obchodní prostory bez udání důvodu?",
        "de": "Wie viele Tage hat ein Verbraucher, um von einem Fernabsatzvertrag oder einem außerhalb von Geschäftsräumen geschlossenen Vertrag ohne Angabe von Gründen zurückzutreten?",
    }),
    ("q00016", "32011L0083", {
        "en": "Within what period must the trader reimburse all payments received from the consumer after being informed of the withdrawal?",
        "cs": "V jaké lhůtě musí obchodník vrátit veškeré platby obdržené od spotřebitele poté, co byl informován o odstoupení od smlouvy?",
        "de": "Innerhalb welcher Frist muss der Unternehmer alle vom Verbraucher erhaltenen Zahlungen erstatten, nachdem er über den Widerruf informiert wurde?",
    }),
    ("q00017", "32019L1024", {
        "en": "What may charges for the re-use of documents held by public sector bodies not exceed?",
        "cs": "Co nesmějí přesáhnout poplatky za opakované použití dokumentů v držení subjektů veřejného sektoru?",
        "de": "Was dürfen Gebühren für die Weiterverwendung von Dokumenten im Besitz öffentlicher Stellen nicht übersteigen?",
    }),
    ("q00018", "32019L1024", {
        "en": "In which formats and at what cost must high-value datasets be made available for re-use?",
        "cs": "V jakých formátech a za jakou cenu musí být datové soubory s vysokou hodnotou zpřístupněny pro opakované použití?",
        "de": "In welchen Formaten und zu welchen Kosten müssen hochwertige Datensätze zur Weiterverwendung bereitgestellt werden?",
    }),
    ("q00019", "32019R1150", {
        "en": "What must a provider of online intermediation services do before restricting or terminating the provision of its services to a given business user?",
        "cs": "Co musí poskytovatel online zprostředkovatelských služeb udělat před omezením nebo ukončením poskytování svých služeb danému podnikatelskému uživateli?",
        "de": "Was muss ein Anbieter von Online-Vermittlungsdiensten tun, bevor er die Erbringung seiner Dienste für einen bestimmten gewerblichen Nutzer einschränkt oder beendet?",
    }),
    ("q00020", "32019R1150", {
        "en": "What transparency about the main parameters determining ranking must providers of online intermediation services give in their terms and conditions?",
        "cs": "Jakou transparentnost ohledně hlavních parametrů určujících pořadí musí poskytovatelé online zprostředkovatelských služeb uvést ve svých obchodních podmínkách?",
        "de": "Welche Transparenz über die Hauptparameter für das Ranking müssen Anbieter von Online-Vermittlungsdiensten in ihren allgemeinen Geschäftsbedingungen bieten?",
    }),
]


def main() -> None:
    with (SMOKE / "queries.jsonl").open("w", encoding="utf-8") as fh:
        for group, celex, variants in QUERIES:
            for lang in ("cs", "en", "de"):
                fh.write(
                    json.dumps(
                        {
                            "query_id": f"{group}_{lang}",
                            "query_group": group,
                            "lang": lang,
                            "text": variants[lang],
                            "source_celex": celex,
                            "origin": "translated",
                            "pivot_lang": "en",
                            "generator": "hand-authored",
                            "translator": "hand-authored",
                            "query_set_version": "smoke",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    with (SMOKE / "qrels.txt").open("w", encoding="utf-8") as fh:
        for group, celex, _ in QUERIES:
            for lang in ("cs", "en", "de"):
                fh.write(f"{group}_{lang} 0 {celex} 1\n")
    print(f"wrote {len(QUERIES) * 3} queries, {len(QUERIES) * 3} qrels")


if __name__ == "__main__":
    main()
