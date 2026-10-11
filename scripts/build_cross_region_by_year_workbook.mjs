import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const runtimeRequire = createRequire(path.join(
  path.dirname(process.execPath), "..", "node_modules", "__artifact_resolver__.cjs"
));
const { SpreadsheetFile, Workbook } = runtimeRequire("@oai/artifact-tool");

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.dirname(scriptDir);
const inputPath = path.join(root, "reports/cross_region_by_year/analysis.json");
const outputDir = path.join(root, "outputs/cross-region-by-year");
const outputPath = path.join(outputDir, "ukraine_alert_directions_2024_2026.xlsx");
const data = JSON.parse(await fs.readFile(inputPath, "utf8"));

const workbook = Workbook.create();
const navy = "#16324F";
const muted = "#52677B";
const bodyFont = { name: "Arial", size: 10, color: "#172B3A" };

function title(sheet, text, subtitle) {
  sheet.showGridLines = false;
  sheet.getRange("A2").values = [[text]];
  sheet.getRange("A2").format.font = { name: "Arial", size: 14, bold: true, color: navy };
  sheet.getRange("A3").values = [[subtitle]];
  sheet.getRange("A3").format.font = { name: "Arial", size: 10, italic: true, color: muted };
  sheet.getRange("A2:P3").format.rowHeight = 21;
}

function writeRows(sheet, startRow, rows, columns) {
  for (let offset = 0; offset < rows.length; offset += 200) {
    const block = rows.slice(offset, offset + 200);
    sheet.getRangeByIndexes(startRow + offset, 0, block.length, columns).values = block;
  }
}

function dataTable(sheet, header, rows, tableName) {
  const lastColumn = String.fromCharCode(64 + header.length);
  sheet.getRange(`A5:${lastColumn}5`).values = [header];
  writeRows(sheet, 5, rows, header.length);
  const lastRow = rows.length + 5;
  const fullRange = sheet.getRange(`A5:${lastColumn}${lastRow}`);
  fullRange.format.font = bodyFont;
  const table = sheet.tables.add(`A5:${lastColumn}${lastRow}`, true, tableName);
  table.style = "TableStyleMedium2";
  table.showFilterButton = true;
  const heading = sheet.getRange(`A5:${lastColumn}5`);
  heading.format.fill = navy;
  heading.format.font = { name: "Arial", size: 10, bold: true, color: "#FFFFFF" };
  heading.format.rowHeight = 28;
  heading.format.wrapText = true;
  sheet.freezePanes.freezeRows(5);
  return lastRow;
}

const coverage = data.coverage.slice().sort((a, b) => a.year - b.year || a.source.localeCompare(b.source));
const pairMap = new Map(data.pairs.map((r) => [
  `${r.origin_region}|${r.target_region}|${r.source}|${r.year}`, r,
]));

const overview = workbook.worksheets.add("Overview");
overview.tabColor = navy;
title(overview, "Ukraine alert directions, 2024–September 2026",
  "All 552 directed region pairs are measured separately for each year and archive.");
overview.getRange("A5:G5").values = [[
  "Year", "Archive", "Coverage end (exclusive UTC)", "Hourly origins", "Directions", "Support passed", "Higher rate",
]];
const coverageRows = coverage.map((r) => [
  r.year, r.source, r.actual_end_exclusive_utc.slice(0, 10), r.hourly_origins,
  r.directed_pairs, r.pairs_passing_support_rule, r.supported_higher_rate_pairs,
]);
overview.getRange("A6:G11").values = coverageRows;
overview.getRange("A5:G5").format.fill = navy;
overview.getRange("A5:G5").format.font = { name: "Arial", size: 10, bold: true, color: "#FFFFFF" };
overview.getRange("A5:G11").format.rowHeight = 22;
overview.getRange("A5:G11").format.font = bodyFont;
overview.getRange("A5:G5").format.font = { name: "Arial", size: 10, bold: true, color: "#FFFFFF" };
overview.getRange("D6:G11").setNumberFormat("#,##0");
overview.getRange("A14").values = [["Repeated higher-rate directions"]];
overview.getRange("A14").format.font = { name: "Arial", size: 12, bold: true, color: navy };
overview.getRange("A15").values = [[
  "These directions pass the support rule and have a higher matched rate in all six source/year comparisons."
]];
overview.getRange("A15").format.font = { name: "Arial", size: 10, italic: true, color: muted };
overview.getRange("A17:D17").values = [["From active region", "To clear region", "Minimum matched ratio", "Minimum onsets"]];
const top12 = data.persistent_higher_rate_pairs.slice(0, 12).map((r) => [
  r.origin_region, r.target_region, r.min_matched_risk_ratio, r.min_active_onsets,
]);
overview.getRange("A18:D29").values = top12;
overview.getRange("A17:D17").format.fill = navy;
overview.getRange("A17:D17").format.font = { name: "Arial", size: 10, bold: true, color: "#FFFFFF" };
overview.getRange("A18:D29").format.font = bodyFont;
overview.getRange("C18:C29").setNumberFormat("0.00");
overview.getRange("D18:D29").setNumberFormat("#,##0");
overview.getRange("A31").values = [[`${data.persistent_higher_rate_pairs.length} repeated directions are listed in full on the Repeated directions sheet.`]];
overview.getRange("A33").values = [["How to read the tables"]];
overview.getRange("A33").format.font = { name: "Arial", size: 12, bold: true, color: navy };
overview.getRange("A34:B37").values = [
  ["Direction", "From region active, to region clear; target alert begins within the next hour."],
  ["Matched comparison", "Calendar quarter, Kyiv local six-hour block and other active regions (0, 1, 2+)."],
  ["Support rule", "At least 200 matched active/clear hours and 10 target onsets."],
  ["Scope", "All alert types. An alert sequence does not establish an aircraft route or causal transfer."],
];
overview.getRange("A39").values = [["Source data"]];
overview.getRange("A39").format.font = { name: "Arial", size: 12, bold: true, color: navy };
overview.getRange("A40:B41").values = [
  ["Official-derived", "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_en.csv"],
  ["Volunteer", "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/volunteer_data_en.csv"],
];
overview.getRange("A43").values = [["The official-derived archive reaches only 6 September 2026. The volunteer archive covers all nine months through 30 September."]];
overview.getRange("A43").format.font = { name: "Arial", size: 10, bold: true, color: "#9A5B00" };
overview.getRange("A:A").format.columnWidth = 24;
overview.getRange("B:B").format.columnWidth = 24;
overview.getRange("C:C").format.columnWidth = 22;
overview.getRange("D:G").format.columnWidth = 16;
overview.getRange("B34:B37").format.columnWidth = 48;
overview.getRange("B40:B41").format.columnWidth = 48;

const repeated = workbook.worksheets.add("Repeated directions");
title(repeated, "Directions repeated across archives and years",
  "Each row has a higher matched rate and meets the support rule in all six comparisons.");
const repeatedHeader = [
  "From active region", "To clear region", "Official 2024", "Volunteer 2024", "Official 2025",
  "Volunteer 2025", "Official 2026*", "Volunteer 2026", "Minimum ratio", "Minimum onsets",
];
const repeatedRows = data.persistent_higher_rate_pairs.map((r) => [
  r.origin_region, r.target_region,
  pairMap.get(`${r.origin_region}|${r.target_region}|official|2024`)?.matched_risk_ratio ?? null,
  pairMap.get(`${r.origin_region}|${r.target_region}|volunteer|2024`)?.matched_risk_ratio ?? null,
  pairMap.get(`${r.origin_region}|${r.target_region}|official|2025`)?.matched_risk_ratio ?? null,
  pairMap.get(`${r.origin_region}|${r.target_region}|volunteer|2025`)?.matched_risk_ratio ?? null,
  pairMap.get(`${r.origin_region}|${r.target_region}|official|2026`)?.matched_risk_ratio ?? null,
  pairMap.get(`${r.origin_region}|${r.target_region}|volunteer|2026`)?.matched_risk_ratio ?? null,
  r.min_matched_risk_ratio, r.min_active_onsets,
]);
const repeatedLast = dataTable(repeated, repeatedHeader, repeatedRows, "RepeatedDirections");
repeated.getRange(`C6:I${repeatedLast}`).setNumberFormat("0.00");
repeated.getRange(`J6:J${repeatedLast}`).setNumberFormat("#,##0");
repeated.getRange("A:A").format.columnWidth = 25;
repeated.getRange("B:B").format.columnWidth = 25;
repeated.getRange("C:J").format.columnWidth = 16;
repeated.getRange(`A${repeatedLast + 2}`).values = [["* Official 2026 coverage ends before 7 September; volunteer 2026 covers through 30 September."]];

const regionSheet = workbook.worksheets.add("By region");
title(regionSheet, "Incoming and outgoing patterns by region",
  "Counts and strongest supported directions are separate for each archive and year.");
const regionHeader = [
  "Year", "Archive", "Region", "Source-active hours", "Target-clear hours", "Outgoing supported",
  "Outgoing higher rate", "Strongest outgoing target", "Outgoing matched RR", "Incoming supported",
  "Incoming higher rate", "Strongest incoming source", "Incoming matched RR", "Coverage end UTC",
];
const regionRows = data.regions.slice().sort((a, b) => a.year - b.year || a.source.localeCompare(b.source) || a.region.localeCompare(b.region)).map((r) => [
  r.year, r.source, r.region, r.source_active_hours, r.target_clear_hours,
  r.supported_outgoing_pairs, r.supported_outgoing_higher_rate_pairs,
  r.top_outgoing_target, r.top_outgoing_matched_risk_ratio,
  r.supported_incoming_pairs, r.supported_incoming_higher_rate_pairs,
  r.top_incoming_source, r.top_incoming_matched_risk_ratio,
  r.coverage_end_exclusive_utc.slice(0, 10),
]);
const regionLast = dataTable(regionSheet, regionHeader, regionRows, "RegionDirections");
regionSheet.getRange(`D6:G${regionLast}`).setNumberFormat("#,##0");
regionSheet.getRange(`I6:I${regionLast}`).setNumberFormat("0.00");
regionSheet.getRange(`J6:K${regionLast}`).setNumberFormat("#,##0");
regionSheet.getRange(`M6:M${regionLast}`).setNumberFormat("0.00");
regionSheet.getRange("A:B").format.columnWidth = 12;
regionSheet.getRange("C:C").format.columnWidth = 25;
regionSheet.getRange("D:G").format.columnWidth = 16;
regionSheet.getRange("H:H").format.columnWidth = 25;
regionSheet.getRange("I:K").format.columnWidth = 16;
regionSheet.getRange("L:L").format.columnWidth = 25;
regionSheet.getRange("M:N").format.columnWidth = 16;

const directionHeader = [
  "Archive", "From active region", "To clear region", "Matched active/clear hours", "Target onsets",
  "Matched comparison hours", "Onset rate with source active", "Matched comparison rate", "Matched risk ratio",
  "Risk difference (pp)", "Reverse direction ratio", "Support rule", "Observed direction",
  "Coverage end UTC", "Crude risk ratio", "All active/clear hours",
];
for (const year of [2024, 2025, 2026]) {
  const sheet = workbook.worksheets.add(`${year} directions`);
  const note = year === 2026
    ? "Official archive: through 6 September. Volunteer archive: through 30 September. Alert order is not a tracked route."
    : "Both archives cover the full calendar year. Alert order is not a tracked route.";
  title(sheet, `${year} directed alert associations`, note);
  const yearRows = data.pairs.filter((r) => r.year === year)
    .sort((a, b) => a.source.localeCompare(b.source) || a.origin_region.localeCompare(b.origin_region) ||
      (b.matched_risk_ratio ?? -1) - (a.matched_risk_ratio ?? -1))
    .map((r) => [
      r.source, r.origin_region, r.target_region,
      r.matched_active_hours, r.matched_active_onsets, r.matched_inactive_hours,
      r.matched_risk_active, r.matched_risk_inactive, r.matched_risk_ratio,
      r.matched_risk_difference_pp, r.reverse_matched_risk_ratio, r.rankable,
      r.association_direction, r.coverage_end_exclusive_utc.slice(0, 10),
      r.crude_risk_ratio, r.origin_active_target_clear_hours,
    ]);
  const last = dataTable(sheet, directionHeader, yearRows, `Directions${year}`);
  sheet.getRange(`D6:F${last}`).setNumberFormat("#,##0");
  sheet.getRange(`G6:H${last}`).setNumberFormat("0.0%");
  sheet.getRange(`I6:I${last}`).setNumberFormat("0.00");
  sheet.getRange(`J6:J${last}`).setNumberFormat("0.00");
  sheet.getRange(`K6:K${last}`).setNumberFormat("0.00");
  sheet.getRange(`O6:O${last}`).setNumberFormat("0.00");
  sheet.getRange(`P6:P${last}`).setNumberFormat("#,##0");
  sheet.getRange("A:A").format.columnWidth = 12;
  sheet.getRange("B:C").format.columnWidth = 25;
  sheet.getRange("D:F").format.columnWidth = 20;
  sheet.getRange("G:K").format.columnWidth = 18;
  sheet.getRange("L:L").format.columnWidth = 14;
  sheet.getRange("M:M").format.columnWidth = 23;
  sheet.getRange("N:N").format.columnWidth = 18;
  sheet.getRange("O:P").format.columnWidth = 18;
}

workbook.recalculate();
const check = await workbook.inspect({
  kind: "region", sheetId: "2024 directions", range: "A5:P8", maxChars: 2500,
  tableMaxRows: 4, tableMaxCols: 16,
});
console.log(check.ndjson);
const errors = await workbook.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 20 }, maxChars: 1000,
});
console.log(errors.ndjson);
for (const sheetName of ["Overview", "Repeated directions", "By region", "2024 directions", "2025 directions", "2026 directions"]) {
  const render = await workbook.render({ sheetName, range: sheetName === "Overview" ? "A1:G44" : "A1:J14", scale: 1.5, format: "png" });
  const safeName = sheetName.toLowerCase().replaceAll(" ", "_");
  await fs.writeFile(path.join(root, `reports/cross_region_by_year/preview_${safeName}.png`),
    new Uint8Array(await render.arrayBuffer()));
}
await fs.mkdir(outputDir, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(outputPath);
