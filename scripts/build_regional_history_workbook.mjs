import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const data = JSON.parse(await fs.readFile(path.join(root, "reports/regional_history_2024_2026/analysis.json"), "utf8"));
const outputDir = path.join(root, "outputs/regional-history-2024-2026");
const outputPath = path.join(outputDir, "ukraine_regional_alert_history_2024_2026.xlsx");
const previewDir = path.join(root, "reports/regional_history_2024_2026");

const wb = Workbook.create();
const dark = "#18364F";
const muted = "#586B7D";
const body = { name: "Arial", size: 10, color: "#203344" };
const head = { name: "Arial", size: 10, color: "#FFFFFF", bold: true };
const periods = ["2024", "2025", "9m 2026"];
const sourceOrder = { volunteer: 0, official_priority: 1 };

function excelUTC(s) {
  return s ? Date.parse(s) / 86400000 + 25569 : null;
}
function title(sh, name, note) {
  sh.showGridLines = false;
  sh.getRange("A2").values = [[name]];
  sh.getRange("A2").format.font = { name: "Arial", size: 15, color: dark, bold: true };
  sh.getRange("A3").values = [[note]];
  sh.getRange("A3").format.font = { name: "Arial", size: 10, color: muted, italic: true };
  sh.getRange("A2:N3").format.rowHeight = 22;
}
function colName(i) {
  let name = "";
  for (let n = i + 1; n > 0; n = Math.floor((n - 1) / 26)) name = String.fromCharCode(65 + (n - 1) % 26) + name;
  return name;
}
function table(sh, headers, rows, tableName) {
  const last = colName(headers.length - 1);
  sh.getRange(`A5:${last}5`).values = [headers];
  for (let offset = 0; offset < rows.length; offset += 150) {
    sh.getRangeByIndexes(5 + offset, 0, Math.min(150, rows.length - offset), headers.length).values = rows.slice(offset, offset + 150);
  }
  const end = rows.length + 5;
  const range = sh.getRange(`A5:${last}${end}`);
  range.format.font = body;
  range.format.rowHeight = 21;
  const t = sh.tables.add(`A5:${last}${end}`, true, tableName);
  t.style = "TableStyleMedium2";
  t.showFilterButton = true;
  const h = sh.getRange(`A5:${last}5`);
  h.format.fill = dark;
  h.format.font = head;
  h.format.rowHeight = 34;
  h.format.wrapText = true;
  h.format.horizontalAlignment = "center";
  sh.freezePanes.freezeRows(5);
  return end;
}

const overview = wb.worksheets.add("Overview");
overview.tabColor = dark;
title(overview, "Ukraine regional alerts, 2024–September 2026",
  "Counts and measured duration by region; threat attribution is provisional for 2026 only.");
const coverage = data.coverage.slice().sort((a, b) => periods.indexOf(a.period) - periods.indexOf(b.period) || sourceOrder[a.source] - sourceOrder[b.source]);
overview.getRange("A5:G5").values = [["Period", "Basis", "Alerts starting", "Usable endings", "Excluded endings", "Long unions", "Regions with starts"]];
overview.getRange("A6:G11").values = coverage.map(r => [r.period, r.source, r.alerts_starting, r.eligible_starting, r.excluded_starting, r.long_unions_starting, r.regions_with_starts]);
overview.getRange("A5:G5").format.fill = dark;
overview.getRange("A5:G5").format.font = head;
overview.getRange("A6:G11").format.font = body;
overview.getRange("A5:G11").format.rowHeight = 23;
overview.getRange("C6:G11").setNumberFormat("#,##0");
overview.getRange("A14").values = [["Highest volunteer region in each period"]];
overview.getRange("A14").format.font = { name: "Arial", size: 12, bold: true, color: dark };
overview.getRange("A16:E16").values = [["Period", "Most alerts", "Alerts", "Most observed alert time", "Days"]];
const headline = periods.map(p => {
  const rows = data.rankings.filter(r => r.period === p && r.source === "volunteer");
  const byCount = rows.slice().sort((a, b) => b.alert_count - a.alert_count || a.region.localeCompare(b.region))[0];
  const byTime = rows.slice().sort((a, b) => b.observed_alert_days - a.observed_alert_days || a.region.localeCompare(b.region))[0];
  return [p, byCount.region, byCount.alert_count, byTime.region, byTime.observed_alert_days];
});
overview.getRange("A17:E19").values = headline;
overview.getRange("A16:E16").format.fill = dark;
overview.getRange("A16:E16").format.font = head;
overview.getRange("A17:E19").format.font = body;
overview.getRange("C17:C19").setNumberFormat("#,##0");
overview.getRange("E17:E19").setNumberFormat("0.00");
overview.getRange("A22").values = [["Threat-type coverage"]];
overview.getRange("A22").format.font = { name: "Arial", size: 12, bold: true, color: dark };
overview.getRange("A24:C24").values = [["Period", "Status", "Selected linked alerts"]];
overview.getRange("A25:C27").values = [["2024", "Unavailable: no collected threat archive", null],
  ["2025", "Unavailable: no collected threat archive", null],
  ["9m 2026", "Provisional candidate associations", data.threat_audit.selected_unique_alerts]];
overview.getRange("A24:C24").format.fill = dark;
overview.getRange("A24:C24").format.font = head;
overview.getRange("A25:C27").format.font = body;
overview.getRange("C27").setNumberFormat("#,##0");
overview.getRange("A30").values = [["Alert days omit uncertain endings. See Source and method for the exact definitions and caveats."]];
overview.getRange("A30").format.font = { name: "Arial", size: 10, italic: true, color: muted };
for (const [column, width] of Object.entries({A: 16, B: 39, C: 25, D: 33, E: 22, F: 18, G: 23})) overview.getRange(`${column}:${column}`).format.columnWidth = width;

const all = wb.worksheets.add("All alert ranks");
title(all, "All-alert regional rankings", "Volunteer is the comparable three-period basis. Official-priority rows provide a separate cross-check.");
const rankingRows = data.rankings.slice().sort((a, b) => periods.indexOf(a.period) - periods.indexOf(b.period) || sourceOrder[a.source] - sourceOrder[b.source] || a.count_rank - b.count_rank || a.region.localeCompare(b.region));
table(all, ["Period", "Basis", "Region", "Count rank", "Alerts starting", "Time rank", "Observed alert days", "Usable endings", "Excluded endings", "Long unions", "Unqualified intervals overlapping period", "Eligible alerts carried over"],
  rankingRows.map(r => [r.period, r.source, r.region, r.count_rank, r.alert_count, r.time_rank, r.observed_alert_days,
    r.duration_eligible_alerts, r.excluded_duration_alerts, r.long_union_alerts,
    r.unqualified_alerts_overlapping_period, r.carried_over_eligible_alerts]), "AllAlertRanks");
all.getRange("E6:F149").setNumberFormat("#,##0");
all.getRange("G6:G149").setNumberFormat("0.00");
all.getRange("H6:L149").setNumberFormat("#,##0");
for (const [column, width] of Object.entries({A: 14, B: 20, C: 28, D: 14, E: 16, F: 13, G: 20, H: 16, I: 17, J: 15, K: 29, L: 24})) all.getRange(`${column}:${column}`).format.columnWidth = width;

const threats = wb.worksheets.add("Threat links 2026");
title(threats, "2026 threat-linked alerts", "Provisional Telegram associations on official-priority canonical episodes; category totals overlap.");
const categoryOrder = ["Any UAV", "Propeller/unspecified attack UAV", "Jet attack UAV", "Reconnaissance UAV", "Any missile", "Ballistic missile", "Cruise/air-launched missile", "Guided aerial bomb", "Tactical aviation"];
const threatRows = data.threat_rankings.slice().sort((a, b) => categoryOrder.indexOf(a.category) - categoryOrder.indexOf(b.category) || a.count_rank - b.count_rank || a.region.localeCompare(b.region));
table(threats, ["Category", "Region", "Count rank", "Linked alerts", "Time rank", "Linked alert days", "Usable endings", "Excluded endings", "Basis"],
  threatRows.map(r => [r.category, r.region, r.count_rank, r.linked_alert_count, r.time_rank, r.linked_alert_days,
    r.linked_alerts_with_usable_end, r.linked_alerts_excluded_duration, "official_priority"]), "ThreatLinkRanks");
threats.getRange("D6:E221").setNumberFormat("#,##0");
threats.getRange("F6:F221").setNumberFormat("0.00");
threats.getRange("G6:H221").setNumberFormat("#,##0");
for (const [column, width] of Object.entries({A: 35, B: 28, C: 14, D: 17, E: 14, F: 19, G: 18, H: 20, I: 20})) threats.getRange(`${column}:${column}`).format.columnWidth = width;

const longs = wb.worksheets.add("Longest alerts");
title(longs, "Longest alert in each region", "UTC start/end times. Recorded maximum includes flagged episodes; eligible maximum has a usable ending.");
const longRows = data.longest.slice().sort((a, b) => periods.indexOf(a.period) - periods.indexOf(b.period) || sourceOrder[a.source] - sourceOrder[b.source] || a.region.localeCompare(b.region));
table(longs, ["Period", "Basis", "Region", "Recorded start UTC", "Recorded end UTC", "Recorded days", "Recorded quality", "Recorded source", "Eligible start UTC", "Eligible end UTC", "Eligible days", "Eligible source"],
  longRows.map(r => [r.period, r.source, r.region,
    excelUTC(r.longest_recorded.start_utc), excelUTC(r.longest_recorded.end_utc), r.longest_recorded.duration_days,
    r.longest_recorded.quality, r.longest_recorded.source_basis,
    excelUTC(r.longest_eligible.start_utc), excelUTC(r.longest_eligible.end_utc), r.longest_eligible.duration_days,
    r.longest_eligible.source_basis]), "LongestAlertRows");
for (const column of ["D", "E", "I", "J"]) longs.getRange(`${column}6:${column}149`).setNumberFormat("yyyy-mm-dd hh:mm:ss");
for (const column of ["F", "K"]) longs.getRange(`${column}6:${column}149`).setNumberFormat("0.000");
for (const [column, width] of Object.entries({A: 14, B: 19, C: 27, D: 24, E: 24, F: 15, G: 47, H: 23, I: 24, J: 24, K: 15, L: 22})) longs.getRange(`${column}:${column}`).format.columnWidth = width;

const average = wb.worksheets.add("Avg duration 2026");
title(average, "Average alert duration, January–September 2026", "Arithmetic mean per alert with an observed, usable ending; days and hours.");
const avgRows = data.averages_2026.slice().sort((a, b) => sourceOrder[a.source] - sourceOrder[b.source] || a.region.localeCompare(b.region));
table(average, ["Basis", "Region", "Eligible alerts", "Excluded endings", "Average days", "Average hours"],
  avgRows.map(r => [r.source, r.region, r.eligible_alerts, r.excluded_duration_alerts,
    r.average_duration_days, r.average_duration_hours]), "AverageDuration2026");
average.getRange("C6:D53").setNumberFormat("#,##0");
average.getRange("E6:E53").setNumberFormat("0.0000");
average.getRange("F6:F53").setNumberFormat("0.00");
for (const [column, width] of Object.entries({A: 20, B: 29, C: 17, D: 20, E: 19, F: 19})) average.getRange(`${column}:${column}`).format.columnWidth = width;

const method = wb.worksheets.add("Source and method");
method.tabColor = "#9BAAB8";
title(method, "Source and method", "Definitions and coverage for the regional alert summary.");
const notes = [
  ["Period", "2024 and 2025 are calendar years. 9m 2026 is January 1 through September 30. Boundaries use UTC."],
  ["Region", "24 represented regions: 23 oblasts plus Kyiv City. The archives do not cover every Ukrainian territory."],
  ["Comparable basis", "Volunteer region-level episodes are used for the headline count, time, longest and average rankings across all three periods."],
  ["Cross-check", "Official-priority canonical episodes combine official-derived records and unmatched volunteer supplements. The official-derived source ends September 7, 2026."],
  ["Alert count", "An alert is one normalized regional episode starting in the period. Missing or estimated endings do not remove its start from the count."],
  ["Observed alert days", "Sum each eligible episode's overlap with the period, in days. Estimated/ambiguous endings, clipped episodes and >7-day unions are excluded."],
  ["Longest recorded", "Maximum nominal episode duration among alerts starting in the period. Check the quality column before interpreting."],
  ["Longest eligible", "Maximum duration among alerts starting in the period with an observed usable ending."],
  ["Average duration", "2026: mean complete eligible duration per alert starting within January–September; days = duration seconds / 86,400."],
  ["Threat years", "2024 and 2025: unavailable; no Telegram threat archive was collected. Blank status means missing classification, never zero alerts."],
  ["Threat selection", "2026: medium regional/time link + corroborated threat event + no manual-review flag. 4,317 of 20,980 canonical episodes qualify."],
  ["Threat meaning", "Count an alert once per linked category. Linked alert days use its full eligible episode duration, not time proven to be caused by the threat."],
  ["Threat versus all alerts", "Threat rankings use official-priority district/region unions. The comparable all-alert rankings use volunteer region-level episodes. Their day totals cannot be compared or subtracted."],
  ["Threat overlap", "A single alert may be linked to UAV and missile reports. Parent categories Any UAV / Any missile overlap their subtypes; never add category totals."],
  ["Source difference", "Volunteer region-level alerts and official district-union alerts describe different alert states. Their durations and counts must not be added."],
  ["Volunteer source", "https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset/blob/main/datasets/volunteer_data_en.csv"],
  ["Official-derived source", "https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset/blob/main/datasets/official_data_en.csv"],
  ["Threat source", "https://t.me/kpszsu"],
];
method.getRange("A5:B5").values = [["Item", "Definition or source"]];
method.getRangeByIndexes(5, 0, notes.length, 2).values = notes;
method.getRange(`A5:B${notes.length + 5}`).format.font = body;
method.getRange("A5:B5").format.fill = dark;
method.getRange("A5:B5").format.font = head;
method.getRange("A:A").format.columnWidth = 29;
method.getRange("B:B").format.columnWidth = 115;
method.getRange(`A6:B${notes.length + 5}`).format.rowHeight = 24;

wb.recalculate();
for (const [sheet, range, name] of [
  ["Overview", "A1:G30", "preview_overview.png"],
  ["All alert ranks", "A1:L11", "preview_all_alert_ranks.png"],
  ["Threat links 2026", "A1:I11", "preview_threat_links.png"],
  ["Longest alerts", "A1:L11", "preview_longest_alerts.png"],
  ["Avg duration 2026", "A1:F11", "preview_average.png"],
  ["Source and method", "A1:B22", "preview_method.png"],
]) {
  const preview = await wb.render({sheetName: sheet, range, scale: 1.25, format: "png"});
  await fs.writeFile(path.join(previewDir, name), new Uint8Array(await preview.arrayBuffer()));
}
const check = await wb.inspect({kind: "region", sheetId: "Overview", range: "A5:G11", maxChars: 2500, tableMaxRows: 7});
console.log(check.ndjson);
const errors = await wb.inspect({kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: {useRegex: true, maxResults: 300}, summary: "final error scan"});
console.log(errors.ndjson);
await fs.mkdir(outputDir, {recursive: true});
const artifact = await SpreadsheetFile.exportXlsx(wb);
await artifact.save(outputPath);
console.log(outputPath);
