import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const [specPath, outputRoot] = process.argv.slice(2);
if (!specPath || !outputRoot) {
  throw new Error("usage: node build_manuscript_workbooks.mjs <workbook_spec.json> <output_root>");
}

const spec = JSON.parse(await fs.readFile(specPath, "utf8"));
const retainedPreviewRoot = process.env.DUODOSE_WORKBOOK_PREVIEW_DIR;
const previewRoot = retainedPreviewRoot || path.join(path.dirname(specPath), ".workbook_previews");
if (!retainedPreviewRoot) {
  await fs.rm(previewRoot, { recursive: true, force: true });
}
await fs.mkdir(previewRoot, { recursive: true });

function columnName(index) {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}

function inferFormat(column) {
  const lower = column.toLowerCase();
  if (lower.includes("fraction") || lower.includes("rate") || lower.includes("auprc") || lower.includes("auroc") || lower.includes("fpr") || lower.includes("recall") || lower.includes("precision") || lower.endsWith("_mean") || lower.endsWith("_std") || lower.endsWith("_sd")) {
    return "0.000";
  }
  if (lower.startsWith("n_") || lower.endsWith("_count") || lower === "seed" || lower === "k") {
    return "#,##0";
  }
  if (lower.includes("seconds")) {
    return "0.00";
  }
  return null;
}

for (const workbookSpec of spec.workbooks) {
  const workbook = Workbook.create();
  for (const sheetSpec of workbookSpec.sheets) {
    const sheet = workbook.worksheets.add(sheetSpec.name);
    sheet.showGridLines = false;
    const rows = [sheetSpec.columns, ...sheetSpec.rows];
    const rowCount = rows.length;
    const columnCount = Math.max(1, sheetSpec.columns.length);
    const endColumn = columnName(columnCount - 1);
    const used = sheet.getRange(`A1:${endColumn}${Math.max(1, rowCount)}`);
    used.values = rows;
    used.format.font = { name: "Arial", size: 10, color: "#202124" };
    used.format.verticalAlignment = "center";
    const header = sheet.getRange(`A1:${endColumn}1`);
    header.format = {
      fill: "#0B6E75",
      font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
      horizontalAlignment: "center",
      verticalAlignment: "center",
      wrapText: true,
      borders: { preset: "outside", style: "thin", color: "#064E55" },
    };
    header.format.rowHeight = 30;
    if (rowCount > 1) {
      const body = sheet.getRange(`A2:${endColumn}${rowCount}`);
      body.format.borders = { preset: "inside", style: "thin", color: "#E1E6EA" };
      body.format.rowHeight = 20;
    }
    for (let index = 0; index < sheetSpec.columns.length; index += 1) {
      const col = columnName(index);
      const format = inferFormat(sheetSpec.columns[index]);
      if (format && rowCount > 1) {
        sheet.getRange(`${col}2:${col}${rowCount}`).format.numberFormat = format;
      }
    }
    used.format.autofitColumns();
    for (let index = 0; index < sheetSpec.columns.length; index += 1) {
      const col = columnName(index);
      const range = sheet.getRange(`${col}1:${col}${Math.max(1, rowCount)}`);
      const lower = sheetSpec.columns[index].toLowerCase();
      if (lower.includes("message") || lower.includes("reason") || lower.includes("source") || lower.includes("feature") || lower.includes("parameter") || lower.includes("note")) {
        range.format.columnWidth = 34;
        range.format.wrapText = true;
      } else {
        range.format.columnWidth = Math.min(18, Math.max(10, sheetSpec.columns[index].length + 2));
      }
    }
    used.format.autofitRows();
    header.format.rowHeight = 30;
    sheet.freezePanes.freezeRows(1);
    const inspect = await workbook.inspect({
      kind: "table",
      range: `${sheetSpec.name}!A1:${endColumn}${Math.min(rowCount, 8)}`,
      include: "values,formulas",
      tableMaxRows: 8,
      tableMaxCols: Math.min(columnCount, 12),
      maxChars: 3000,
    });
    if (!inspect.ndjson) {
      throw new Error(`empty inspection result for ${workbookSpec.output} / ${sheetSpec.name}`);
    }
    const preview = await workbook.render({ sheetName: sheetSpec.name, autoCrop: "all", scale: 1, format: "png" });
    const previewBytes = new Uint8Array(await preview.arrayBuffer());
    if (previewBytes.length === 0) {
      throw new Error(`empty preview for ${workbookSpec.output} / ${sheetSpec.name}`);
    }
    await fs.writeFile(path.join(previewRoot, `${path.basename(workbookSpec.output, ".xlsx")}__${sheetSpec.name}.png`), previewBytes);
  }
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "manuscript workbook formula error scan",
    maxChars: 2000,
  });
  if (errors.ndjson && /#REF!|#DIV\/0!|#VALUE!|#NAME\?|#N\/A/.test(errors.ndjson)) {
    throw new Error(`formula error token found in ${workbookSpec.output}`);
  }
  const destination = path.join(outputRoot, workbookSpec.output);
  await fs.mkdir(path.dirname(destination), { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(destination);
}

if (!retainedPreviewRoot) {
  await fs.rm(previewRoot, { recursive: true, force: true });
}
