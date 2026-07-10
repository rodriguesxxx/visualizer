import { readFile, writeFile } from "node:fs/promises";

const xlsxPath = new URL("../public/reports/analysis-history-template.xlsx", import.meta.url);
const csvPath = new URL("../public/reports/analysis-history-template.csv", import.meta.url);
const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

const writeUint16 = (target, offset, value) => {
  target[offset] = value & 0xff;
  target[offset + 1] = (value >>> 8) & 0xff;
};

const writeUint32 = (target, offset, value) => {
  target[offset] = value & 0xff;
  target[offset + 1] = (value >>> 8) & 0xff;
  target[offset + 2] = (value >>> 16) & 0xff;
  target[offset + 3] = (value >>> 24) & 0xff;
};

const concatBytes = (parts) => {
  const output = new Uint8Array(parts.reduce((total, part) => total + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.length;
  }
  return output;
};

const crc32Table = Array.from({ length: 256 }, (_, index) => {
  let value = index;
  for (let bit = 0; bit < 8; bit += 1) {
    value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
  }
  return value >>> 0;
});

const crc32 = (bytes) => {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = crc32Table[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
};

const readStoredZip = (input) => {
  const bytes = new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let endRecordOffset = -1;

  for (let offset = bytes.length - 22; offset >= Math.max(0, bytes.length - 65_557); offset -= 1) {
    if (view.getUint32(offset, true) === 0x06054b50) {
      endRecordOffset = offset;
      break;
    }
  }
  if (endRecordOffset < 0) throw new Error("Invalid XLSX template.");

  const entries = view.getUint16(endRecordOffset + 10, true);
  let centralOffset = view.getUint32(endRecordOffset + 16, true);
  const files = new Map();

  for (let index = 0; index < entries; index += 1) {
    if (view.getUint16(centralOffset + 10, true) !== 0) throw new Error("XLSX entries must be stored without compression.");
    const size = view.getUint32(centralOffset + 20, true);
    const filenameLength = view.getUint16(centralOffset + 28, true);
    const extraLength = view.getUint16(centralOffset + 30, true);
    const commentLength = view.getUint16(centralOffset + 32, true);
    const localOffset = view.getUint32(centralOffset + 42, true);
    const filename = textDecoder.decode(bytes.slice(centralOffset + 46, centralOffset + 46 + filenameLength));
    const localFilenameLength = view.getUint16(localOffset + 26, true);
    const localExtraLength = view.getUint16(localOffset + 28, true);
    const dataOffset = localOffset + 30 + localFilenameLength + localExtraLength;
    files.set(filename, bytes.slice(dataOffset, dataOffset + size));
    centralOffset += 46 + filenameLength + extraLength + commentLength;
  }

  return files;
};

const createStoredZip = (files) => {
  const localParts = [];
  const centralParts = [];
  let localOffset = 0;

  for (const [filename, content] of files) {
    const nameBytes = textEncoder.encode(filename);
    const checksum = crc32(content);
    const localHeader = new Uint8Array(30 + nameBytes.length);
    writeUint32(localHeader, 0, 0x04034b50);
    writeUint16(localHeader, 4, 20);
    writeUint32(localHeader, 14, checksum);
    writeUint32(localHeader, 18, content.length);
    writeUint32(localHeader, 22, content.length);
    writeUint16(localHeader, 26, nameBytes.length);
    localHeader.set(nameBytes, 30);
    localParts.push(localHeader, content);

    const centralHeader = new Uint8Array(46 + nameBytes.length);
    writeUint32(centralHeader, 0, 0x02014b50);
    writeUint16(centralHeader, 4, 20);
    writeUint16(centralHeader, 6, 20);
    writeUint32(centralHeader, 16, checksum);
    writeUint32(centralHeader, 20, content.length);
    writeUint32(centralHeader, 24, content.length);
    writeUint16(centralHeader, 28, nameBytes.length);
    writeUint32(centralHeader, 42, localOffset);
    centralHeader.set(nameBytes, 46);
    centralParts.push(centralHeader);
    localOffset += localHeader.length + content.length;
  }

  const centralDirectory = concatBytes(centralParts);
  const endRecord = new Uint8Array(22);
  writeUint32(endRecord, 0, 0x06054b50);
  writeUint16(endRecord, 8, files.size);
  writeUint16(endRecord, 10, files.size);
  writeUint32(endRecord, 12, centralDirectory.length);
  writeUint32(endRecord, 16, localOffset);
  return concatBytes([...localParts, centralDirectory, endRecord]);
};

const columnNameFromIndex = (index) => {
  let dividend = index + 1;
  let name = "";
  while (dividend > 0) {
    const modulo = (dividend - 1) % 26;
    name = String.fromCharCode(65 + modulo) + name;
    dividend = Math.floor((dividend - modulo) / 26);
  }
  return name;
};

const csv = await readFile(csvPath, "utf8");
const headers = csv.trim().split(/\r?\n/, 1)[0].split(",");
const cells = headers
  .map((header, index) => {
    const reference = `${columnNameFromIndex(index)}1`;
    return `<c r="${reference}" t="inlineStr"><is><t>${header}</t></is></c>`;
  })
  .join("");
const lastColumn = columnNameFromIndex(headers.length - 1);
const worksheet = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="A1:${lastColumn}1" />
  <sheetViews><sheetView workbookViewId="0" /></sheetViews>
  <sheetFormatPr defaultRowHeight="15" />
  <cols>
    <col min="1" max="3" width="24" customWidth="1" />
    <col min="4" max="4" width="32" customWidth="1" />
    <col min="5" max="${headers.length}" width="18" customWidth="1" />
  </cols>
  <sheetData><row r="1">${cells}</row></sheetData>
</worksheet>`;

const files = readStoredZip(await readFile(xlsxPath));
files.set("xl/worksheets/sheet1.xml", textEncoder.encode(worksheet));
await writeFile(xlsxPath, createStoredZip(files));
