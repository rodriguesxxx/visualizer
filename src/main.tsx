import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  Boxes,
  Camera,
  ChevronDown,
  CheckCircle2,
  CircleDot,
  Download,
  Eye,
  FileSpreadsheet,
  History,
  ImageUp,
  Images,
  Layers3,
  Leaf,
  Minus,
  Plus,
  ScanLine,
  SlidersVertical,
  Sparkles,
  Trash2
} from "lucide-react";
import "./styles.css";

type ClassName = "folha" | "fruto";
type ViewMode = "masks" | "boxes" | "compare";

type Detection = {
  id: string;
  className: ClassName;
  confidence: number;
  bbox: { x: number; y: number; w: number; h: number };
  mask: string;
  area: number;
  lab: { l: number; a: number; b: number };
};

type TrainingMetric = {
  epoch: number;
  map50: number;
  precision: number;
  recall: number;
  loss: number;
};

type TrainingArtifact = {
  id: string;
  file: string;
  url: string;
  sizeBytes: number;
};

type AnalysisHistoryItem = {
  id: string;
  plantId: string;
  analyzedAt: string;
  imageName: string;
  imageSizeBytes: number;
  imageSize: string;
  imageType: string;
  leafCount: number;
  fruitCount: number;
  totalDetections: number;
  averageConfidence: number;
  model: string;
  latencyMs: number | null;
};

type ApiDetection = {
  id: string;
  className: string;
  confidence: number;
  bbox: { x: number; y: number; w: number; h: number };
  mask: Array<{ x: number; y: number }>;
  area: number;
  lab: { l: number; a: number; b: number };
};

type AnalyzeResponse = {
  model: string;
  latencyMs: number;
  image: {
    originalDataUrl: string;
    annotatedDataUrl: string;
  };
  counts: Record<string, number>;
  detections: ApiDetection[];
};

type ApiTrainingMetric = {
  epoch: number;
  map50Mask: number;
  precisionMask: number;
  recallMask: number;
  boxLoss: number;
  segLoss: number;
  classLoss: number;
};

type ApiModelInfo = {
  name?: string;
};

type Size = {
  width: number;
  height: number;
};

type Pan = {
  x: number;
  y: number;
};

const viteEnv = (import.meta as unknown as { env?: { BASE_URL?: string; VITE_MODEL_API_URL?: string } }).env;
const appBaseUrl = viteEnv?.BASE_URL ?? "/";
const publicAssetBaseUrl = appBaseUrl.endsWith("/") ? appBaseUrl : `${appBaseUrl}/`;
const API_BASE = (viteEnv?.VITE_MODEL_API_URL ?? "http://localhost:8000").replace(/\/$/, "");
const inferenceTimeoutMs = 90_000;
const maxConvertedImageSide = 1280;
const jpegQuality = 0.82;
const apiCacheTtlMs = 12 * 60 * 60 * 1000;
const apiCachePrefix = "plant-ai-api-cache";
const artifactImageCacheName = "plant-ai-artifact-images-v1";
const analysisHistoryStorageKey = "plant-ai-analysis-history-v1";
const maxAnalysisHistoryItems = 100;
const analysisReportTemplateCsvUrl = `${publicAssetBaseUrl}reports/analysis-history-template.csv`;
const analysisReportTemplateXlsxUrl = `${publicAssetBaseUrl}reports/analysis-history-template.xlsx`;
const analysisReportHeaders = [
  "plant_id",
  "analysis_id",
  "analyzed_at",
  "image_name",
  "image_size_bytes",
  "image_size",
  "image_type",
  "leaf_count",
  "fruit_count",
  "total_detections",
  "average_confidence",
  "model",
  "latency_ms"
];
const spreadsheetXmlNamespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
const pendingJsonRequests = new Map<string, Promise<unknown>>();
const pendingArtifactImageRequests = new Map<string, Promise<string>>();

const initialDetections: Detection[] = [
  {
    id: "leaf-01",
    className: "folha",
    confidence: 0.94,
    bbox: { x: 13, y: 18, w: 32, h: 20 },
    mask: "15% 26%, 23% 17%, 39% 18%, 45% 29%, 31% 37%, 18% 34%",
    area: 10.8,
    lab: { l: 42, a: -34, b: 28 }
  },
  {
    id: "leaf-02",
    className: "folha",
    confidence: 0.91,
    bbox: { x: 52, y: 28, w: 27, h: 24 },
    mask: "54% 39%, 61% 28%, 75% 30%, 79% 44%, 68% 52%, 56% 49%",
    area: 9.2,
    lab: { l: 47, a: -31, b: 33 }
  },
  {
    id: "fruit-01",
    className: "fruto",
    confidence: 0.88,
    bbox: { x: 39, y: 43, w: 14, h: 29 },
    mask: "45% 43%, 53% 52%, 50% 68%, 43% 73%, 38% 61%, 40% 49%",
    area: 4.1,
    lab: { l: 39, a: 45, b: 31 }
  }
];

const initialTrainingMetrics: TrainingMetric[] = [
  { epoch: 1, map50: 0.18, precision: 0.32, recall: 0.21, loss: 2.9 },
  { epoch: 10, map50: 0.41, precision: 0.58, recall: 0.43, loss: 1.74 },
  { epoch: 20, map50: 0.57, precision: 0.69, recall: 0.56, loss: 1.18 },
  { epoch: 30, map50: 0.67, precision: 0.75, recall: 0.62, loss: 0.91 },
  { epoch: 40, map50: 0.73, precision: 0.8, recall: 0.68, loss: 0.76 },
  { epoch: 50, map50: 0.78, precision: 0.84, recall: 0.72, loss: 0.64 }
];

const classStyle = {
  folha: { label: "Folha", color: "#42f58d", soft: "rgba(66, 245, 141, .25)" },
  fruto: { label: "Fruto", color: "#ff5c7a", soft: "rgba(255, 92, 122, .25)" }
};

const imageAccept = "image/*,.heic,.heif";
const heicExtensions = [".heic", ".heif"];

const isSupportedImageFile = (file: File) => {
  const filename = file.name.toLowerCase();
  return file.type.startsWith("image/") || heicExtensions.some((extension) => filename.endsWith(extension));
};

const isHeicImageFile = (file: File) => {
  const filename = file.name.toLowerCase();
  return file.type === "image/heic" || file.type === "image/heif" || heicExtensions.some((extension) => filename.endsWith(extension));
};

const getJpegFilename = (filename: string) => filename.replace(/\.(heic|heif)$/i, ".jpg") || "imagem.jpg";

const loadImageElement = (url: string) =>
  new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Não foi possível preparar o JPG convertido."));
    image.src = url;
  });

const canvasToJpegBlob = (canvas: HTMLCanvasElement) =>
  new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob);
          return;
        }

        reject(new Error("Não foi possível gerar o JPG convertido."));
      },
      "image/jpeg",
      jpegQuality
    );
  });

const normalizeJpegBlob = async (blob: Blob, filename: string) => {
  let source: ImageBitmap | HTMLImageElement | null = null;
  let objectUrl: string | null = null;

  try {
    try {
      source = await createImageBitmap(blob);
    } catch (error) {
      objectUrl = URL.createObjectURL(blob);
      source = await loadImageElement(objectUrl);
    }

    const scale = Math.min(1, maxConvertedImageSide / Math.max(source.width, source.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(source.width * scale));
    canvas.height = Math.max(1, Math.round(source.height * scale));

    const context = canvas.getContext("2d");
    if (!context) throw new Error("Não foi possível preparar o JPG convertido.");

    context.drawImage(source, 0, 0, canvas.width, canvas.height);
    const jpegBlob = await canvasToJpegBlob(canvas);
    return new File([jpegBlob], filename, { type: "image/jpeg" });
  } finally {
    if (source && "close" in source && typeof source.close === "function") source.close();
    if (objectUrl) URL.revokeObjectURL(objectUrl);
  }
};

const convertHeicToJpeg = async (file: File) => {
  try {
    const { default: heic2any } = await import("heic2any");
    const result = await heic2any({ blob: file, toType: "image/jpeg", quality: jpegQuality });
    const jpegBlob = Array.isArray(result) ? result[0] : result;
    if (!jpegBlob) throw new Error("HEIC sem imagem válida.");
    return normalizeJpegBlob(jpegBlob, getJpegFilename(file.name));
  } catch (error) {
    throw new Error("Não foi possível converter HEIC para JPG.");
  }
};

const prepareImageFileForUpload = (file: File) => {
  if (!isHeicImageFile(file)) return Promise.resolve(file);
  return convertHeicToJpeg(file);
};

const readJsonCache = <T,>(key: string): T | null => {
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return null;
    const cached = JSON.parse(raw) as { timestamp: number; value: T };
    if (Date.now() - cached.timestamp > apiCacheTtlMs) return null;
    return cached.value;
  } catch (error) {
    return null;
  }
};

const writeJsonCache = <T,>(key: string, value: T) => {
  try {
    window.localStorage.setItem(key, JSON.stringify({ timestamp: Date.now(), value }));
  } catch (error) {
    // Storage can fail in private mode or when quota is exceeded; network fallback remains valid.
  }
};

const fetchCachedJson = async <T,>(cacheKey: string, url: string) => {
  const key = `${apiCachePrefix}:${cacheKey}`;
  const cached = readJsonCache<T>(key);
  if (cached) return cached;

  const pending = pendingJsonRequests.get(key) as Promise<T> | undefined;
  if (pending) return pending;

  const request = fetch(url, { cache: "force-cache" })
    .then(async (response) => {
      if (!response.ok) throw new Error(`Falha ao carregar ${cacheKey}`);
      const value = (await response.json()) as T;
      writeJsonCache(key, value);
      return value;
    })
    .finally(() => pendingJsonRequests.delete(key));

  pendingJsonRequests.set(key, request);
  return request;
};

const getCachedArtifactImageUrl = async (url: string) => {
  if (!("caches" in window)) return url;

  const pending = pendingArtifactImageRequests.get(url);
  if (pending) return pending;

  const request = (async () => {
    try {
      const cache = await caches.open(artifactImageCacheName);
      const cached = await cache.match(url);
      if (cached) return URL.createObjectURL(await cached.blob());

      const response = await fetch(url, { cache: "force-cache" });
      if (!response.ok) return url;

      await cache.put(url, response.clone());
      return URL.createObjectURL(await response.blob());
    } catch (error) {
      return url;
    }
  })().finally(() => pendingArtifactImageRequests.delete(url));

  pendingArtifactImageRequests.set(url, request);
  return request;
};

const formatFileSize = (bytes: number) => {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
};

const generateRandomId = (prefix: string) => {
  if (window.crypto?.getRandomValues) {
    const bytes = new Uint8Array(6);
    window.crypto.getRandomValues(bytes);
    return `${prefix}-${Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("").toUpperCase()}`;
  }

  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`.toUpperCase();
};

const countDetectionsByClass = (items: Detection[]) =>
  items.reduce(
    (acc, detection) => {
      acc[detection.className] += 1;
      return acc;
    },
    { folha: 0, fruto: 0 }
  );

const getAverageConfidence = (items: Detection[]) =>
  items.length ? items.reduce((total, detection) => total + detection.confidence, 0) / items.length : 0;

const isAnalysisHistoryItem = (item: unknown): item is AnalysisHistoryItem => {
  if (!item || typeof item !== "object") return false;
  const candidate = item as Partial<AnalysisHistoryItem>;
  return (
    typeof candidate.id === "string" &&
    typeof candidate.plantId === "string" &&
    typeof candidate.analyzedAt === "string" &&
    typeof candidate.imageName === "string" &&
    typeof candidate.imageSizeBytes === "number" &&
    typeof candidate.leafCount === "number" &&
    typeof candidate.fruitCount === "number"
  );
};

const readAnalysisHistory = () => {
  try {
    const raw = window.localStorage.getItem(analysisHistoryStorageKey);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isAnalysisHistoryItem).slice(0, maxAnalysisHistoryItems);
  } catch (error) {
    return [];
  }
};

const writeAnalysisHistory = (items: AnalysisHistoryItem[]) => {
  try {
    window.localStorage.setItem(analysisHistoryStorageKey, JSON.stringify(items.slice(0, maxAnalysisHistoryItems)));
    return true;
  } catch (error) {
    return false;
  }
};

const createAnalysisHistoryItem = ({
  file,
  detections,
  model,
  latencyMs
}: {
  file: File;
  detections: Detection[];
  model: string;
  latencyMs: number | null;
}): AnalysisHistoryItem => {
  const counts = countDetectionsByClass(detections);

  return {
    id: generateRandomId("ANL"),
    plantId: generateRandomId("PLT"),
    analyzedAt: new Date().toISOString(),
    imageName: file.name,
    imageSizeBytes: file.size,
    imageSize: formatFileSize(file.size),
    imageType: file.type || "desconhecido",
    leafCount: counts.folha,
    fruitCount: counts.fruto,
    totalDetections: detections.length,
    averageConfidence: getAverageConfidence(detections),
    model,
    latencyMs
  };
};

const formatDateTime = (value: string) => {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return new Intl.DateTimeFormat("pt-BR", {
    dateStyle: "short",
    timeStyle: "short"
  }).format(date);
};

const reportColumnGetters: Record<string, (item: AnalysisHistoryItem) => string | number | null> = {
  plant_id: (item) => item.plantId,
  analysis_id: (item) => item.id,
  analyzed_at: (item) => item.analyzedAt,
  image_name: (item) => item.imageName,
  image_size_bytes: (item) => item.imageSizeBytes,
  image_size: (item) => item.imageSize,
  image_type: (item) => item.imageType,
  leaf_count: (item) => item.leafCount,
  fruit_count: (item) => item.fruitCount,
  total_detections: (item) => item.totalDetections,
  average_confidence: (item) => Number(item.averageConfidence.toFixed(4)),
  model: (item) => item.model,
  latency_ms: (item) => item.latencyMs
};

const normalizeReportHeaderColumn = (column: string) => column.trim().replace(/^"|"$/g, "");
const getReportColumns = (header: string) => header.split(",").map(normalizeReportHeaderColumn);
const getReportValue = (column: string, item: AnalysisHistoryItem) => reportColumnGetters[column]?.(item) ?? "";

const escapeCsvValue = (value: string | number | null | undefined) => {
  const text = value === null || value === undefined ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, "\"\"")}"` : text;
};

const encodeUtf8 = (value: string) => new TextEncoder().encode(value);
const decodeUtf8 = (value: Uint8Array) => new TextDecoder().decode(value);

const xmlEscape = (value: string | number | null | undefined) =>
  String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

const crc32Table = Array.from({ length: 256 }, (_, index) => {
  let value = index;
  for (let bit = 0; bit < 8; bit += 1) {
    value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
  }
  return value >>> 0;
});

const crc32 = (bytes: Uint8Array) => {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc = crc32Table[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
};

const concatBytes = (parts: Uint8Array[]) => {
  const totalLength = parts.reduce((total, part) => total + part.length, 0);
  const output = new Uint8Array(totalLength);
  let offset = 0;

  for (const part of parts) {
    output.set(part, offset);
    offset += part.length;
  }

  return output;
};

const writeUint16 = (target: Uint8Array, offset: number, value: number) => {
  target[offset] = value & 0xff;
  target[offset + 1] = (value >>> 8) & 0xff;
};

const writeUint32 = (target: Uint8Array, offset: number, value: number) => {
  target[offset] = value & 0xff;
  target[offset + 1] = (value >>> 8) & 0xff;
  target[offset + 2] = (value >>> 16) & 0xff;
  target[offset + 3] = (value >>> 24) & 0xff;
};

const createStoredZip = (files: Map<string, Uint8Array>) => {
  const localParts: Uint8Array[] = [];
  const centralParts: Uint8Array[] = [];
  let localOffset = 0;

  for (const [filename, content] of files) {
    const nameBytes = encodeUtf8(filename);
    const checksum = crc32(content);
    const localHeader = new Uint8Array(30 + nameBytes.length);
    writeUint32(localHeader, 0, 0x04034b50);
    writeUint16(localHeader, 4, 20);
    writeUint16(localHeader, 6, 0);
    writeUint16(localHeader, 8, 0);
    writeUint16(localHeader, 10, 0);
    writeUint16(localHeader, 12, 0);
    writeUint32(localHeader, 14, checksum);
    writeUint32(localHeader, 18, content.length);
    writeUint32(localHeader, 22, content.length);
    writeUint16(localHeader, 26, nameBytes.length);
    writeUint16(localHeader, 28, 0);
    localHeader.set(nameBytes, 30);
    localParts.push(localHeader, content);

    const centralHeader = new Uint8Array(46 + nameBytes.length);
    writeUint32(centralHeader, 0, 0x02014b50);
    writeUint16(centralHeader, 4, 20);
    writeUint16(centralHeader, 6, 20);
    writeUint16(centralHeader, 8, 0);
    writeUint16(centralHeader, 10, 0);
    writeUint16(centralHeader, 12, 0);
    writeUint16(centralHeader, 14, 0);
    writeUint32(centralHeader, 16, checksum);
    writeUint32(centralHeader, 20, content.length);
    writeUint32(centralHeader, 24, content.length);
    writeUint16(centralHeader, 28, nameBytes.length);
    writeUint16(centralHeader, 30, 0);
    writeUint16(centralHeader, 32, 0);
    writeUint16(centralHeader, 34, 0);
    writeUint16(centralHeader, 36, 0);
    writeUint32(centralHeader, 38, 0);
    writeUint32(centralHeader, 42, localOffset);
    centralHeader.set(nameBytes, 46);
    centralParts.push(centralHeader);

    localOffset += localHeader.length + content.length;
  }

  const centralDirectory = concatBytes(centralParts);
  const endRecord = new Uint8Array(22);
  writeUint32(endRecord, 0, 0x06054b50);
  writeUint16(endRecord, 4, 0);
  writeUint16(endRecord, 6, 0);
  writeUint16(endRecord, 8, files.size);
  writeUint16(endRecord, 10, files.size);
  writeUint32(endRecord, 12, centralDirectory.length);
  writeUint32(endRecord, 16, localOffset);
  writeUint16(endRecord, 20, 0);

  return concatBytes([...localParts, centralDirectory, endRecord]);
};

const readStoredZip = (buffer: ArrayBuffer) => {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  let endRecordOffset = -1;
  const minOffset = Math.max(0, bytes.length - 65_557);

  for (let offset = bytes.length - 22; offset >= minOffset; offset -= 1) {
    if (view.getUint32(offset, true) === 0x06054b50) {
      endRecordOffset = offset;
      break;
    }
  }

  if (endRecordOffset < 0) throw new Error("Template Excel inválido.");

  const entries = view.getUint16(endRecordOffset + 10, true);
  let centralOffset = view.getUint32(endRecordOffset + 16, true);
  const files = new Map<string, Uint8Array>();

  for (let index = 0; index < entries; index += 1) {
    if (view.getUint32(centralOffset, true) !== 0x02014b50) throw new Error("Template Excel inválido.");

    const compressionMethod = view.getUint16(centralOffset + 10, true);
    const compressedSize = view.getUint32(centralOffset + 20, true);
    const filenameLength = view.getUint16(centralOffset + 28, true);
    const extraLength = view.getUint16(centralOffset + 30, true);
    const commentLength = view.getUint16(centralOffset + 32, true);
    const localHeaderOffset = view.getUint32(centralOffset + 42, true);
    const filename = decodeUtf8(bytes.slice(centralOffset + 46, centralOffset + 46 + filenameLength));
    const localFilenameLength = view.getUint16(localHeaderOffset + 26, true);
    const localExtraLength = view.getUint16(localHeaderOffset + 28, true);
    const dataOffset = localHeaderOffset + 30 + localFilenameLength + localExtraLength;

    if (compressionMethod !== 0) throw new Error("A base Excel precisa estar em ZIP sem compressão.");

    files.set(filename, bytes.slice(dataOffset, dataOffset + compressedSize));
    centralOffset += 46 + filenameLength + extraLength + commentLength;
  }

  return files;
};

const columnNameFromIndex = (index: number) => {
  let dividend = index + 1;
  let columnName = "";

  while (dividend > 0) {
    const modulo = (dividend - 1) % 26;
    columnName = String.fromCharCode(65 + modulo) + columnName;
    dividend = Math.floor((dividend - modulo) / 26);
  }

  return columnName;
};

const buildWorksheetCellXml = (value: string | number | null | undefined, cellReference: string) => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return `<c r="${cellReference}"><v>${value}</v></c>`;
  }

  return `<c r="${cellReference}" t="inlineStr"><is><t>${xmlEscape(value)}</t></is></c>`;
};

const buildAnalysisWorksheetXml = (items: AnalysisHistoryItem[]) => {
  const rows = [
    analysisReportHeaders,
    ...items.map((item) => analysisReportHeaders.map((column) => getReportValue(column, item)))
  ];
  const lastColumn = columnNameFromIndex(analysisReportHeaders.length - 1);
  const sheetData = rows
    .map((row, rowIndex) => {
      const rowNumber = rowIndex + 1;
      const cells = row
        .map((value, columnIndex) => buildWorksheetCellXml(value, `${columnNameFromIndex(columnIndex)}${rowNumber}`))
        .join("");
      return `<row r="${rowNumber}">${cells}</row>`;
    })
    .join("");

  return `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="${spreadsheetXmlNamespace}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="A1:${lastColumn}${rows.length}" />
  <sheetViews><sheetView workbookViewId="0" /></sheetViews>
  <sheetFormatPr defaultRowHeight="15" />
  <cols>
    <col min="1" max="3" width="24" customWidth="1" />
    <col min="4" max="4" width="32" customWidth="1" />
    <col min="5" max="13" width="18" customWidth="1" />
  </cols>
  <sheetData>${sheetData}</sheetData>
</worksheet>`;
};

const createAnalysisWorkbookFiles = (worksheetXml: string) =>
  new Map<string, Uint8Array>([
    [
      "[Content_Types].xml",
      encodeUtf8(`<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />
  <Default Extension="xml" ContentType="application/xml" />
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml" />
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml" />
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml" />
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml" />
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml" />
</Types>`)
    ],
    [
      "_rels/.rels",
      encodeUtf8(`<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml" />
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml" />
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml" />
</Relationships>`)
    ],
    [
      "docProps/core.xml",
      encodeUtf8(`<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>Plant.AI Visualizer</dc:creator>
  <cp:lastModifiedBy>Plant.AI Visualizer</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-07-08T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">2026-07-08T00:00:00Z</dcterms:modified>
</cp:coreProperties>`)
    ],
    [
      "docProps/app.xml",
      encodeUtf8(`<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Plant.AI Visualizer</Application>
</Properties>`)
    ],
    [
      "xl/workbook.xml",
      encodeUtf8(`<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="${spreadsheetXmlNamespace}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Historico" sheetId="1" r:id="rId1" /></sheets>
</workbook>`)
    ],
    [
      "xl/_rels/workbook.xml.rels",
      encodeUtf8(`<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml" />
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml" />
</Relationships>`)
    ],
    [
      "xl/styles.xml",
      encodeUtf8(`<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="${spreadsheetXmlNamespace}">
  <fonts count="1"><font><sz val="11" /><name val="Calibri" /></font></fonts>
  <fills count="1"><fill><patternFill patternType="none" /></fill></fills>
  <borders count="1"><border><left /><right /><top /><bottom /><diagonal /></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" /></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" /></cellXfs>
</styleSheet>`)
    ],
    ["xl/worksheets/sheet1.xml", encodeUtf8(worksheetXml)]
  ]);

const getAnalysisReportTemplateHeader = async () => {
  try {
    const response = await fetch(analysisReportTemplateCsvUrl, { cache: "no-cache" });
    if (response.ok) {
      const template = await response.text();
      const header = template.split(/\r?\n/).find((line) => line.trim());
      if (header) return header.trim();
    }
  } catch (error) {
    // The exported report can still be generated with the built-in header.
  }

  return analysisReportHeaders.join(",");
};

const buildAnalysisHistoryCsv = async (items: AnalysisHistoryItem[]) => {
  const header = await getAnalysisReportTemplateHeader();
  const columns = getReportColumns(header);
  const rows = items.map((item) =>
    columns.map((column) => escapeCsvValue(getReportValue(column, item))).join(",")
  );

  return `\uFEFF${[header, ...rows].join("\r\n")}\r\n`;
};

const downloadBlobFile = (filename: string, blob: Blob) => {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
};

const exportAnalysisHistoryCsv = async (items: AnalysisHistoryItem[]) => {
  const csv = await buildAnalysisHistoryCsv(items);
  const date = new Date().toISOString().slice(0, 10);
  downloadBlobFile(`plant-ai-historico-analises-${date}.csv`, new Blob([csv], { type: "text/csv;charset=utf-8" }));
};

const getAnalysisWorkbookTemplateFiles = async () => {
  try {
    const response = await fetch(analysisReportTemplateXlsxUrl, { cache: "no-cache" });
    if (response.ok) return readStoredZip(await response.arrayBuffer());
  } catch (error) {
    // The Excel report can still be generated from the built-in workbook structure.
  }

  return createAnalysisWorkbookFiles(buildAnalysisWorksheetXml([]));
};

const buildAnalysisHistoryXlsx = async (items: AnalysisHistoryItem[]) => {
  const files = await getAnalysisWorkbookTemplateFiles();
  files.set("xl/worksheets/sheet1.xml", encodeUtf8(buildAnalysisWorksheetXml(items)));
  return createStoredZip(files);
};

const exportAnalysisHistoryXlsx = async (items: AnalysisHistoryItem[]) => {
  const workbook = await buildAnalysisHistoryXlsx(items);
  const date = new Date().toISOString().slice(0, 10);
  downloadBlobFile(
    `plant-ai-historico-analises-${date}.xlsx`,
    new Blob([workbook], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" })
  );
};

const analyzeImageRequest = async (form: FormData) => {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), inferenceTimeoutMs);

  try {
    return await fetch(`${API_BASE}/api/v1/inference/analyze?confidence=0.25&iou=0.7`, {
      method: "POST",
      body: form,
      signal: controller.signal
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("A API demorou mais de 90s para responder à inferência.");
    }

    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
};

const getAnalyzeErrorMessage = (error: unknown) => {
  const errorMessage = error instanceof Error ? error.message : "";
  if (errorMessage.includes("converter HEIC")) return errorMessage;
  if (errorMessage.includes("demorou mais de 90s")) return errorMessage;
  return `Não foi possível analisar a imagem. Verifique se a API está acessível em ${API_BASE}.`;
};

function App() {
  const beforeCardRef = useRef<HTMLDivElement>(null);
  const afterCardRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ pointerId: number; startX: number; startY: number; origin: Pan } | null>(null);
  const dropDepthRef = useRef(0);
  const [viewMode, setViewMode] = useState<ViewMode>("compare");
  const [selectedClass, setSelectedClass] = useState<ClassName | "todas">("todas");
  const [detections, setDetections] = useState<Detection[]>(initialDetections);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetric[]>(initialTrainingMetrics);
  const [artifacts, setArtifacts] = useState<TrainingArtifact[]>([]);
  const [imageSrc, setImageSrc] = useState(
    "https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?auto=format&fit=crop&w=1400&q=82"
  );
  const [analysisModel, setAnalysisModel] = useState("Plant.AI YOLOv8-seg v1");
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [apiStatus, setApiStatus] = useState<"loading" | "online" | "offline">("loading");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [maskFocus, setMaskFocus] = useState(58);
  const [analysisZoom, setAnalysisZoom] = useState(1);
  const [analysisPan, setAnalysisPan] = useState<Pan>({ x: 0, y: 0 });
  const [imageAspect, setImageAspect] = useState(1.55);
  const [isDraggingImage, setIsDraggingImage] = useState(false);
  const [isInstancesExpanded, setIsInstancesExpanded] = useState(false);
  const [isArtifactsExpanded, setIsArtifactsExpanded] = useState(false);
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisHistoryItem[]>(readAnalysisHistory);
  const [isExportingHistory, setIsExportingHistory] = useState(false);
  const [historyMessage, setHistoryMessage] = useState("");
  const beforeSize = useElementSize(beforeCardRef);
  const afterSize = useElementSize(afterCardRef);
  const [message, setMessage] = useState("Carregando metadados do treino v1...");

  useEffect(() => {
    const loadApiData = async () => {
      try {
        const [info, metrics, artifactPayload] = await Promise.all([
          fetchCachedJson<ApiModelInfo>("model-info", `${API_BASE}/api/v1/model/info`),
          fetchCachedJson<{ series: ApiTrainingMetric[] }>("training-metrics", `${API_BASE}/api/v1/training/metrics`),
          fetchCachedJson<{ artifacts: TrainingArtifact[] }>("training-artifacts", `${API_BASE}/api/v1/training/artifacts`)
        ]);

        setAnalysisModel(info.name ?? "Plant.AI YOLOv8-seg v1");
        setTrainingMetrics(metrics.series.map(toTrainingMetric));
        setArtifacts(artifactPayload.artifacts);
        setApiStatus("online");
        setMessage(`API conectada em ${API_BASE}`);
      } catch (error) {
        setApiStatus("offline");
        setMessage("API do modelo indisponível. Exibindo dados demonstrativos.");
      }
    };

    void loadApiData();
  }, []);

  const filteredDetections = useMemo(
    () =>
      selectedClass === "todas"
        ? detections
        : detections.filter((detection) => detection.className === selectedClass),
    [detections, selectedClass]
  );

  const counts = useMemo(() => countDetectionsByClass(detections), [detections]);

  const confidence = getAverageConfidence(detections);
  const totalArea = detections.reduce((total, detection) => total + detection.area, 0);
  const visibleDetections = isAnalyzing ? [] : filteredDetections;
  const lastMetric = trainingMetrics[trainingMetrics.length - 1];
  const focusDim = 0.12 + (maskFocus / 100) * 0.62;
  const focusBoost = 1.04 + (maskFocus / 100) * 0.3;
  const beforeSurface = getContainedSurface(beforeSize, imageAspect);
  const analysisSurface = getContainedSurface(afterSize, imageAspect);
  const updateAnalysisZoom = (delta: number) =>
    setAnalysisZoom((current) => {
      const next = Math.min(2.5, Math.max(1, Number((current + delta).toFixed(2))));
      setAnalysisPan((pan) => clampPan(pan, next, analysisSurface));
      return next;
    });
  const updateContrastFromPointer = (event: React.PointerEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (bounds.bottom - event.clientY) / bounds.height));
    setMaskFocus(Math.round(ratio * 100));
  };
  const updateImageAspect = (event: React.SyntheticEvent<HTMLImageElement>) => {
    const { naturalWidth, naturalHeight } = event.currentTarget;
    if (naturalWidth && naturalHeight) setImageAspect(naturalWidth / naturalHeight);
  };
  const startPan = (event: React.PointerEvent<HTMLDivElement>) => {
    if (analysisZoom <= 1) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      origin: analysisPan
    };
  };
  const movePan = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const next = {
      x: drag.origin.x + event.clientX - drag.startX,
      y: drag.origin.y + event.clientY - drag.startY
    };
    setAnalysisPan(clampPan(next, analysisZoom, analysisSurface));
  };
  const stopPan = (event: React.PointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.pointerId === event.pointerId) dragRef.current = null;
  };

  useEffect(() => {
    setAnalysisPan((pan) => clampPan(pan, analysisZoom, analysisSurface));
  }, [analysisSurface.width, analysisSurface.height, analysisZoom]);

  const addAnalysisHistoryItem = (item: AnalysisHistoryItem) => {
    const next = [item, ...analysisHistory].slice(0, maxAnalysisHistoryItems);
    const wasSaved = writeAnalysisHistory(next);
    setAnalysisHistory(next);
    setHistoryMessage(wasSaved ? `Análise ${item.plantId} salva no navegador.` : "Análise registrada apenas nesta sessão.");
  };

  const handleExportHistory = async (format: "csv" | "xlsx") => {
    if (!analysisHistory.length || isExportingHistory) return;

    setIsExportingHistory(true);
    setHistoryMessage(format === "xlsx" ? "Montando Excel..." : "Montando CSV...");

    try {
      if (format === "xlsx") {
        await exportAnalysisHistoryXlsx(analysisHistory);
      } else {
        await exportAnalysisHistoryCsv(analysisHistory);
      }
      setHistoryMessage(`${format === "xlsx" ? "Excel" : "CSV"} gerado com ${analysisHistory.length} análises.`);
    } catch (error) {
      setHistoryMessage(`Não foi possível gerar o ${format === "xlsx" ? "Excel" : "CSV"}.`);
    } finally {
      setIsExportingHistory(false);
    }
  };

  const clearAnalysisHistory = () => {
    if (!analysisHistory.length) return;
    const shouldClear = window.confirm("Limpar o histórico de análises salvo neste navegador?");
    if (!shouldClear) return;

    setAnalysisHistory([]);
    writeAnalysisHistory([]);
    setHistoryMessage("Histórico local limpo.");
  };

  const analyzeFile = async (file: File) => {
    if (!isSupportedImageFile(file)) {
      setMessage("Arquivo ignorado. Arraste ou selecione uma imagem JPG, PNG, HEIC ou HEIF.");
      return;
    }

    setIsAnalyzing(true);
    setMessage(isHeicImageFile(file) ? "Convertendo HEIC para JPG..." : "Enviando imagem para inferência na API do modelo...");

    try {
      const uploadFile = await prepareImageFileForUpload(file);
      const localPreview = URL.createObjectURL(uploadFile);
      setImageSrc(localPreview);
      setMessage(`Enviando ${isHeicImageFile(file) ? "JPG convertido" : "imagem"} (${formatFileSize(uploadFile.size)}) para inferência...`);

      const form = new FormData();
      form.append("file", uploadFile, uploadFile.name);

      const response = await analyzeImageRequest(form);

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const payload = (await response.json()) as AnalyzeResponse;
      const nextDetections = payload.detections.map(toDetection).filter(Boolean) as Detection[];
      const historyItem = createAnalysisHistoryItem({
        file,
        detections: nextDetections,
        model: payload.model,
        latencyMs: payload.latencyMs
      });

      setImageSrc(payload.image.originalDataUrl);
      setDetections(nextDetections);
      setLatencyMs(payload.latencyMs);
      setAnalysisModel(payload.model);
      setApiStatus("online");
      addAnalysisHistoryItem(historyItem);
      setMessage(`Inferência concluída em ${(payload.latencyMs / 1000).toFixed(2)}s`);
    } catch (error) {
      setApiStatus("offline");
      setMessage(getAnalyzeErrorMessage(error));
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleImageInput = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) void analyzeFile(file);
    event.currentTarget.value = "";
  };

  const startImageDrag = (event: React.DragEvent<HTMLElement>) => {
    event.preventDefault();
    if (isAnalyzing) return;
    dropDepthRef.current += 1;
    event.dataTransfer.dropEffect = "copy";
    setIsDraggingImage(true);
  };

  const keepImageDrag = (event: React.DragEvent<HTMLElement>) => {
    event.preventDefault();
    if (!isAnalyzing) event.dataTransfer.dropEffect = "copy";
  };

  const stopImageDrag = (event: React.DragEvent<HTMLElement>) => {
    event.preventDefault();
    dropDepthRef.current = Math.max(0, dropDepthRef.current - 1);
    if (dropDepthRef.current === 0) setIsDraggingImage(false);
  };

  const dropImage = (event: React.DragEvent<HTMLElement>) => {
    event.preventDefault();
    dropDepthRef.current = 0;
    setIsDraggingImage(false);

    if (isAnalyzing) return;
    const file = Array.from(event.dataTransfer.files).find(isSupportedImageFile);
    if (file) {
      void analyzeFile(file);
      return;
    }

    setMessage("Nenhuma imagem encontrada no arquivo arrastado.");
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Plant.AI Visualizer</p>
          <h1>Análise visual YOLOv8-seg v1</h1>
        </div>
        <div className={`live-pill ${apiStatus}`}>
          <span />
          {apiStatus === "online" ? "API conectada" : apiStatus === "loading" ? "Conectando API" : "Modo demo"}
        </div>
      </header>

      <section className="dashboard-grid">
        <article
          className={`analysis-stage ${isDraggingImage ? "is-dragging-image" : ""}`}
          onDragEnter={startImageDrag}
          onDragOver={keepImageDrag}
          onDragLeave={stopImageDrag}
          onDrop={dropImage}
        >
          <div className="drop-overlay">
            <ImageUp size={28} />
            <span>Solte a imagem para analisar</span>
          </div>
          <div className="panel-header">
            <div>
              <p className="eyebrow">Antes e depois</p>
              <h2>Segmentação de folhas e frutos</h2>
              <p className="api-message">{message}</p>
            </div>
            <div className={`upload-actions ${isAnalyzing ? "disabled" : ""}`} aria-label="Selecionar imagem">
              <label className="upload-button camera-action">
                <Camera size={17} />
                {isAnalyzing ? "Analisando" : "Câmera"}
                <input disabled={isAnalyzing} type="file" accept={imageAccept} capture="environment" onChange={handleImageInput} />
              </label>
              <label className="upload-button secondary">
                <Images size={17} />
                Galeria
                <input disabled={isAnalyzing} type="file" accept={imageAccept} onChange={handleImageInput} />
              </label>
            </div>
          </div>

          <div className={`image-workbench ${viewMode}`}>
            <div className="image-card before" ref={beforeCardRef}>
              <div className="image-surface" style={beforeSurface}>
                <img src={imageSrc} alt="Imagem original da planta" onLoad={updateImageAspect} />
              </div>
              <span>Original</span>
            </div>
            <div className="image-card after" ref={afterCardRef}>
              <div
                className={`analysis-viewport ${analysisZoom > 1 ? "is-pannable" : ""}`}
                onPointerDown={startPan}
                onPointerMove={movePan}
                onPointerUp={stopPan}
                onPointerCancel={stopPan}
              >
                <div
                  className="analysis-canvas"
                  style={
                    {
                      ...analysisSurface,
                      "--mask-dim": focusDim,
                      "--mask-boost": focusBoost,
                      transform: `translate(${analysisPan.x}px, ${analysisPan.y}px) scale(${analysisZoom})`
                    } as React.CSSProperties
                  }
                >
                  <img src={imageSrc} alt="Imagem com análise de segmentação" onLoad={updateImageAspect} />
                  {viewMode !== "boxes" && visibleDetections.length > 0 && (
                    <MaskFocusOverlay detections={visibleDetections} imageSrc={imageSrc} />
                  )}
                  <div className="scan-grid" />
                  <div className="scan-beam" />
                  {visibleDetections.map((detection) => (
                    <DetectionLayer key={detection.id} detection={detection} mode={viewMode} />
                  ))}
                </div>
              </div>
              <div className="map-controls">
                <div className="zoom-control" aria-label="Zoom da imagem analisada">
                  <button
                    type="button"
                    onClick={() => updateAnalysisZoom(0.25)}
                    disabled={analysisZoom >= 2.5}
                    aria-label="Aumentar zoom"
                  >
                    <Plus size={18} />
                  </button>
                  <button
                    type="button"
                    onClick={() => updateAnalysisZoom(-0.25)}
                    disabled={analysisZoom <= 1}
                    aria-label="Diminuir zoom"
                  >
                    <Minus size={18} />
                  </button>
                </div>
                <label className="contrast-control" aria-label="Foco das máscaras">
                  <SlidersVertical size={15} />
                  <div
                    className="contrast-slider"
                    role="slider"
                    tabIndex={0}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={maskFocus}
                    onPointerDown={(event) => {
                      event.currentTarget.setPointerCapture(event.pointerId);
                      updateContrastFromPointer(event);
                    }}
                    onPointerMove={(event) => {
                      if (event.buttons === 1) updateContrastFromPointer(event);
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "ArrowUp" || event.key === "ArrowRight") {
                        setMaskFocus((current) => Math.min(100, current + 2));
                      }
                      if (event.key === "ArrowDown" || event.key === "ArrowLeft") {
                        setMaskFocus((current) => Math.max(0, current - 2));
                      }
                    }}
                  >
                    <span className="contrast-track">
                      <span className="contrast-fill" style={{ height: `${maskFocus}%` }} />
                      <span className="contrast-thumb" style={{ bottom: `${maskFocus}%` }} />
                    </span>
                  </div>
                  <strong>{maskFocus}%</strong>
                </label>
              </div>
              <span>{isAnalyzing ? "Processando" : "Analisada"}</span>
            </div>
          </div>

          <div className="control-strip">
            <Segmented
              value={viewMode}
              options={[
                { value: "compare", label: "Comparar", icon: <Eye size={16} /> },
                { value: "masks", label: "Máscaras", icon: <Layers3 size={16} /> },
                { value: "boxes", label: "Boxes", icon: <Boxes size={16} /> }
              ]}
              onChange={setViewMode}
            />
            <Segmented
              value={selectedClass}
              options={[
                { value: "todas", label: "Todas", icon: <Sparkles size={16} /> },
                { value: "folha", label: "Folhas", icon: <Leaf size={16} /> },
                { value: "fruto", label: "Frutos", icon: <CircleDot size={16} /> }
              ]}
              onChange={setSelectedClass}
            />
          </div>
        </article>

        <aside className="side-stack">
          <div className="metric-grid">
            <Metric label="Folhas" value={isAnalyzing ? "-" : counts.folha} color="#42f58d" />
            <Metric label="Frutos" value={isAnalyzing ? "-" : counts.fruto} color="#ff5c7a" />
            <Metric label="Confiança" value={isAnalyzing ? "-" : `${Math.round(confidence * 100)}%`} color="#69c8ff" />
            <Metric label="Área seg." value={isAnalyzing ? "-" : `${totalArea.toFixed(1)}%`} color="#f2d66d" />
          </div>

          <section className="data-panel">
            <PanelTitle icon={<BarChart3 size={18} />} title="Contagem por classe" />
            {isAnalyzing ? (
              <div className="processing-state">Em processamento</div>
            ) : (
              <BarChart
                data={[
                  { label: "Folhas", value: counts.folha, color: "#42f58d" },
                  { label: "Frutos", value: counts.fruto, color: "#ff5c7a" }
                ]}
              />
            )}
          </section>

          <section className="data-panel">
            <PanelTitle icon={<Activity size={18} />} title="Treinamento v1" />
            <LineChart data={trainingMetrics} />
            <div className="training-note">
              <CheckCircle2 size={16} />
              <span>
                {lastMetric
                  ? `mAP50 máscara ${(lastMetric.map50 * 100).toFixed(1)}%, precisão ${(
                      lastMetric.precision * 100
                    ).toFixed(1)}%, recall ${(lastMetric.recall * 100).toFixed(1)}%.`
                  : "Aguardando métricas do treino."}
              </span>
            </div>
          </section>

          <AnalysisHistoryPanel
            history={analysisHistory}
            isExporting={isExportingHistory}
            status={historyMessage}
            templateCsvUrl={analysisReportTemplateCsvUrl}
            templateXlsxUrl={analysisReportTemplateXlsxUrl}
            onClear={clearAnalysisHistory}
            onExportCsv={() => handleExportHistory("csv")}
            onExportXlsx={() => handleExportHistory("xlsx")}
          />
        </aside>
      </section>

      <section className="bottom-grid">
        <article className="data-panel instances-panel">
          <div className="panel-title-row">
            <PanelTitle icon={<ScanLine size={18} />} title="Instâncias detectadas" />
            <div className="panel-actions">
              <span>{isAnalyzing ? "Processando" : `${detections.length} instâncias`}</span>
              <button
                aria-expanded={isInstancesExpanded}
                aria-label={isInstancesExpanded ? "Recolher instâncias detectadas" : "Expandir instâncias detectadas"}
                className="expand-button"
                onClick={() => setIsInstancesExpanded((current) => !current)}
                type="button"
              >
                <ChevronDown size={17} />
              </button>
            </div>
          </div>
          {isInstancesExpanded && (
            isAnalyzing ? (
              <div className="processing-state">Em processamento</div>
            ) : (
              <div className="instance-list">
                {detections.map((detection) => (
                  <InstanceRow key={detection.id} detection={detection} />
                ))}
              </div>
            )
          )}
        </article>

        <article className="data-panel">
          <div className="panel-title-row">
            <PanelTitle icon={<Layers3 size={18} />} title="Modelo e artefatos" />
            <div className="panel-actions">
              <button
                aria-expanded={isArtifactsExpanded}
                aria-label={isArtifactsExpanded ? "Recolher modelo e artefatos" : "Expandir modelo e artefatos"}
                className="expand-button"
                onClick={() => setIsArtifactsExpanded((current) => !current)}
                type="button"
              >
                <ChevronDown size={17} />
              </button>
            </div>
          </div>
          {isArtifactsExpanded && (
            <>
              <div className="model-summary">
                <strong>{analysisModel}</strong>
                <span>{latencyMs ? `Última inferência: ${(latencyMs / 1000).toFixed(2)}s` : "Aguardando upload"}</span>
              </div>
              <ArtifactGrid artifacts={artifacts} />
            </>
          )}
        </article>
      </section>
    </main>
  );
}

function AnalysisHistoryPanel({
  history,
  isExporting,
  status,
  templateCsvUrl,
  templateXlsxUrl,
  onClear,
  onExportCsv,
  onExportXlsx
}: {
  history: AnalysisHistoryItem[];
  isExporting: boolean;
  status: string;
  templateCsvUrl: string;
  templateXlsxUrl: string;
  onClear: () => void;
  onExportCsv: () => void;
  onExportXlsx: () => void;
}) {
  const savedLabel = `${history.length} ${history.length === 1 ? "salva" : "salvas"}`;
  const latestItems = history.slice(0, 3);

  return (
    <section className="data-panel history-panel">
      <div className="panel-title-row">
        <PanelTitle icon={<History size={18} />} title="Histórico de análises" />
        <span className="history-count">{savedLabel}</span>
      </div>

      <div className="history-actions">
        <button className="report-action primary" disabled={!history.length || isExporting} onClick={onExportXlsx} type="button">
          <FileSpreadsheet size={16} />
          {isExporting ? "Gerando" : "Exportar Excel"}
        </button>
        <button className="report-action" disabled={!history.length || isExporting} onClick={onExportCsv} type="button">
          <Download size={16} />
          CSV
        </button>
        <button
          aria-label="Limpar histórico local"
          className="report-icon-button"
          disabled={!history.length}
          onClick={onClear}
          type="button"
        >
          <Trash2 size={16} />
        </button>
      </div>

      <div className="template-actions">
        <a className="template-link" href={templateXlsxUrl} rel="noreferrer" target="_blank">
          <FileSpreadsheet size={15} />
          Base Excel
        </a>
        <a className="template-link" href={templateCsvUrl} rel="noreferrer" target="_blank">
          <FileSpreadsheet size={16} />
          Base CSV
        </a>
      </div>

      {status && <p className="history-status">{status}</p>}

      {latestItems.length ? (
        <div className="history-list" aria-label="Últimas análises salvas">
          {latestItems.map((item) => (
            <div className="history-item" key={item.id}>
              <strong>{item.plantId}</strong>
              <span title={item.imageName}>{item.imageName}</span>
              <small>
                {formatDateTime(item.analyzedAt)} | {item.leafCount} folhas | {item.fruitCount} frutos
              </small>
            </div>
          ))}
        </div>
      ) : (
        <p className="empty-state">Nenhuma análise salva.</p>
      )}
    </section>
  );
}

function toTrainingMetric(item: ApiTrainingMetric): TrainingMetric {
  return {
    epoch: item.epoch,
    map50: item.map50Mask,
    precision: item.precisionMask,
    recall: item.recallMask,
    loss: item.boxLoss + item.segLoss + item.classLoss
  };
}

function toDetection(item: ApiDetection): Detection | null {
  if (item.className !== "folha" && item.className !== "fruto") {
    return null;
  }

  return {
    id: item.id,
    className: item.className,
    confidence: item.confidence,
    bbox: item.bbox,
    mask: item.mask.map((point) => `${point.x}% ${point.y}%`).join(", "),
    area: item.area,
    lab: item.lab
  };
}

function absoluteApiUrl(path: string) {
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}

function DetectionLayer({ detection, mode }: { detection: Detection; mode: ViewMode }) {
  const theme = classStyle[detection.className];
  const points = toSvgPoints(detection.mask);

  return (
    <div className="detection-layer">
      {mode !== "boxes" && detection.mask && (
        <svg className="mask-outline" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          <polygon points={points} style={{ "--mask-color": theme.color } as React.CSSProperties} />
        </svg>
      )}
      {mode !== "masks" && (
        <div
          className="box"
          style={{
            borderColor: theme.color,
            color: theme.color,
            left: `${detection.bbox.x}%`,
            top: `${detection.bbox.y}%`,
            width: `${detection.bbox.w}%`,
            height: `${detection.bbox.h}%`
          }}
        >
          <span>{Math.round(detection.confidence * 100)}%</span>
        </div>
      )}
    </div>
  );
}

function MaskFocusOverlay({ detections, imageSrc }: { detections: Detection[]; imageSrc: string }) {
  return (
    <div className="mask-focus-layer" aria-hidden="true">
      <div className="mask-dim" />
      {detections.map((detection) => (
        <div
          className="mask-spotlight"
          key={detection.id}
          style={{
            clipPath: `polygon(${detection.mask})`,
            backgroundImage: `url("${imageSrc}")`
          }}
        />
      ))}
    </div>
  );
}

function toSvgPoints(mask: string) {
  return mask
    .split(",")
    .map((point) =>
      point
        .trim()
        .replace(/%/g, "")
        .split(/\s+/)
        .join(",")
    )
    .join(" ");
}

function useElementSize<T extends HTMLElement>(ref: React.RefObject<T>) {
  const [size, setSize] = useState<Size>({ width: 0, height: 0 });

  useEffect(() => {
    if (!ref.current) return;
    const element = ref.current;
    const updateSize = () => {
      const bounds = element.getBoundingClientRect();
      setSize({ width: bounds.width, height: bounds.height });
    };
    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref]);

  return size;
}

function getContainedSurface(size: Size, aspect: number): React.CSSProperties {
  if (!size.width || !size.height) return {};

  const containerAspect = size.width / size.height;
  if (containerAspect > aspect) {
    return { width: size.height * aspect, height: size.height };
  }

  return { width: size.width, height: size.width / aspect };
}

function clampPan(pan: Pan, zoom: number, surface: React.CSSProperties): Pan {
  if (zoom <= 1 || typeof surface.width !== "number" || typeof surface.height !== "number") {
    return { x: 0, y: 0 };
  }

  const maxX = (surface.width * (zoom - 1)) / 2;
  const maxY = (surface.height * (zoom - 1)) / 2;
  return {
    x: Math.min(maxX, Math.max(-maxX, pan.x)),
    y: Math.min(maxY, Math.max(-maxY, pan.y))
  };
}

function Metric({ label, value, color }: { label: string; value: string | number; color: string }) {
  return (
    <div className="metric-card" style={{ "--accent": color } as React.CSSProperties}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function BarChart({ data }: { data: Array<{ label: string; value: number; color: string }> }) {
  const max = Math.max(1, ...data.map((item) => item.value));
  return (
    <div className="bar-chart">
      {data.map((item) => (
        <div className="bar-row" key={item.label}>
          <span>{item.label}</span>
          <div className="bar-track">
            <div
              className="bar-fill"
              style={{
                width: `${(item.value / max) * 100}%`,
                background: item.color
              }}
            />
          </div>
          <strong>{item.value}</strong>
        </div>
      ))}
    </div>
  );
}

function LineChart({ data }: { data: TrainingMetric[] }) {
  const width = 420;
  const height = 170;
  const padding = 18;
  const safeData = data.length > 1 ? data : initialTrainingMetrics;
  const maxLoss = Math.max(1, ...safeData.map((item) => item.loss));
  const points = safeData
    .map((item, index) => {
      const x = padding + (index / (safeData.length - 1)) * (width - padding * 2);
      const y = height - padding - item.map50 * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");
  const lossPoints = safeData
    .map((item, index) => {
      const x = padding + (index / (safeData.length - 1)) * (width - padding * 2);
      const normalizedLoss = Math.min(item.loss / maxLoss, 1);
      const y = padding + normalizedLoss * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg className="line-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Curva de treino v1">
      <polyline points={lossPoints} fill="none" stroke="#ffb86b" strokeWidth="3" opacity=".75" />
      <polyline points={points} fill="none" stroke="#42f58d" strokeWidth="4" />
      {safeData.map((item, index) => {
        const x = padding + (index / (safeData.length - 1)) * (width - padding * 2);
        const y = height - padding - item.map50 * (height - padding * 2);
        return <circle key={`${item.epoch}-${index}`} cx={x} cy={y} r="4" fill="#f5fff9" />;
      })}
    </svg>
  );
}

function ArtifactGrid({ artifacts }: { artifacts: TrainingArtifact[] }) {
  const selected = artifacts.filter((artifact) =>
    ["results", "confusion_matrix_normalized", "mask_pr_curve", "mask_f1_curve"].includes(artifact.id)
  );

  if (!selected.length) {
    return <p className="empty-state">Artefatos do treino serão exibidos quando a API estiver conectada.</p>;
  }

  return (
    <div className="artifact-grid">
      {selected.map((artifact) => (
        <CachedArtifactLink artifact={artifact} key={artifact.id} />
      ))}
    </div>
  );
}

function CachedArtifactLink({ artifact }: { artifact: TrainingArtifact }) {
  const originalUrl = absoluteApiUrl(artifact.url);
  const [imageUrl, setImageUrl] = useState(originalUrl);

  useEffect(() => {
    let objectUrl: string | null = null;
    let isActive = true;

    void getCachedArtifactImageUrl(originalUrl).then((cachedUrl) => {
      if (!isActive) {
        if (cachedUrl.startsWith("blob:")) URL.revokeObjectURL(cachedUrl);
        return;
      }

      objectUrl = cachedUrl.startsWith("blob:") ? cachedUrl : null;
      setImageUrl(cachedUrl);
    });

    return () => {
      isActive = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [originalUrl]);

  return (
    <a href={originalUrl} target="_blank" rel="noreferrer">
      <img src={imageUrl} alt={artifact.file} loading="lazy" />
      <span>{artifact.id.replace(/_/g, " ")}</span>
    </a>
  );
}

function InstanceRow({ detection }: { detection: Detection }) {
  const theme = classStyle[detection.className];

  return (
    <div className="instance-row">
      <div className="chip" style={{ borderColor: theme.color }}>
        <span style={{ background: theme.color }} />
        {theme.label}
      </div>
      <strong>{Math.round(detection.confidence * 100)}%</strong>
      <small>
        L* {detection.lab.l} | a* {detection.lab.a} | b* {detection.lab.b}
      </small>
    </div>
  );
}

function PanelTitle({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <div className="panel-title">
      {icon}
      <h3>{title}</h3>
    </div>
  );
}

function Segmented<T extends string>({
  value,
  options,
  onChange
}: {
  value: T;
  options: Array<{ value: T; label: string; icon: React.ReactNode }>;
  onChange: (value: T) => void;
}) {
  return (
    <div className="segmented">
      {options.map((option) => (
        <button
          className={option.value === value ? "active" : ""}
          key={option.value}
          onClick={() => onChange(option.value)}
          type="button"
        >
          {option.icon}
          {option.label}
        </button>
      ))}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
