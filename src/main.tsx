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
  Ruler,
  ScanLine,
  SlidersVertical,
  Sparkles,
  Trash2
} from "lucide-react";
import "./styles.css";

type ClassName = "folha" | "fruto" | "marcador_aruco" | "planta_inteira";
type ViewMode = "masks" | "boxes" | "compare";
type AnalysisSource = "visualizer" | "api";

type Detection = {
  id: string;
  className: ClassName;
  confidence: number;
  bbox: { x: number; y: number; w: number; h: number };
  mask: string;
  area: number;
  lab: { l: number; a: number; b: number };
  parentId?: string;
  state?: string;
  size?: {
    imagePercent?: { width?: number; height?: number; area?: number };
    pixel?: { unit?: string; width?: number; height?: number; area?: number };
    real?: { unit?: string; width?: number; height?: number; areaMm2?: number };
  };
};

type ScaleMeasurement = {
  calibrated?: boolean;
  reason?: string;
  dictionary?: string;
  configuredMarkerSizeMm?: number | null;
  markersDetected?: number;
  markerIds?: number[];
  reference?: string;
  yoloMarkersDetected?: number;
  averageSidePixels?: number;
  medianSidePixels?: number;
  mmPerPixel?: number;
  pixelsPerMm?: number;
  markers?: Array<{
    id?: number;
    averageSidePixels?: number;
    corners?: Array<{ x?: number; y?: number }>;
  }>;
};

type HeightMeasurement = {
  unit?: string;
  calibrated?: boolean;
  count?: number;
  average?: number;
  max?: number;
  pixel?: { unit?: string; average?: number; max?: number };
  real?: { unit?: string; average?: number; max?: number };
};

type SizeMeasurement = {
  unit?: string;
  calibrated?: boolean;
  count?: number;
  averageWidth?: number;
  averageHeight?: number;
  averageArea?: number;
  pixel?: { unit?: string; averageWidth?: number; averageHeight?: number; averageArea?: number };
  real?: {
    unit?: string;
    averageWidth?: number;
    averageHeight?: number;
    averageAreaMm2?: number;
  };
};

type InferenceMeasurements = {
  scale?: ScaleMeasurement;
  plantHeight?: HeightMeasurement;
  fruitSize?: SizeMeasurement;
  fruitStates?: { method?: string; counts?: Record<string, number> };
  leafSize?: SizeMeasurement;
  [key: string]: unknown;
};

type DetectionMeasurementRecord = {
  id: string;
  className: ClassName;
  confidence: number;
  area: number;
  lab: Detection["lab"];
  parentId?: string;
  state?: string;
  size?: Detection["size"];
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
  plantCount?: number;
  markerCount?: number;
  totalDetections: number;
  averageConfidence: number;
  model: string;
  datasetVersion?: string;
  trainingMap50Mask?: number;
  trainingPrecisionMask?: number;
  trainingRecallMask?: number;
  latencyMs: number | null;
  imageWidth?: number;
  imageHeight?: number;
  originalImageWidth?: number;
  originalImageHeight?: number;
  measurements?: InferenceMeasurements;
  detectionMeasurements?: DetectionMeasurementRecord[];
};

type ApiDetection = {
  id: string;
  className: string;
  confidence: number;
  bbox: { x: number; y: number; w: number; h: number };
  mask?: Array<{ x: number; y: number }>;
  area?: number;
  lab?: { l: number; a: number; b: number };
  parentId?: string;
  state?: string;
  size?: Detection["size"];
};

type AnalyzeResponse = {
  model: string | ApiModelInfo;
  filename?: string;
  latencyMs: number;
  image: {
    width?: number;
    height?: number;
    originalWidth?: number;
    originalHeight?: number;
    originalDataUrl?: string;
    annotatedDataUrl?: string;
  };
  counts: Record<string, number>;
  detections: ApiDetection[];
  thresholds?: InferenceParams;
  measurements?: InferenceMeasurements;
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

type ApiClassInfo = {
  id: number;
  name: string;
};

type ApiModelInfo = {
  id?: string;
  name?: string;
  task?: string;
  datasetVersion?: string;
  baseModel?: string;
  classes?: Array<string | ApiClassInfo>;
  weights?: {
    current?: { exists?: boolean; path?: string; sizeBytes?: number };
    last?: { exists?: boolean; path?: string; sizeBytes?: number };
    loadedFrom?: string;
  };
  inference?: {
    configuredImgSize?: number;
    defaultConfidence?: number;
    defaultIou?: number;
    runtime?: { name?: string; device?: string; imgsz?: number; maxDetections?: number };
    pipeline?: string;
    topLevelClasses?: string[];
    childClasses?: string[];
  };
  training?: {
    epochs?: number;
    completedEpochs?: number;
    batch?: number;
    device?: string;
    optimizer?: string;
    pretrained?: boolean;
    finalMetrics?: ApiTrainingMetric;
  };
};

type ApiTrainingSummary = {
  datasetVersion?: string;
  final?: ApiTrainingMetric;
};

type Size = {
  width: number;
  height: number;
};

type Pan = {
  x: number;
  y: number;
};

type InferenceParams = {
  confidence: number;
  iou: number;
};

const viteEnv = (import.meta as unknown as { env?: { BASE_URL?: string; VITE_MODEL_API_URL?: string } }).env;
const appBaseUrl = viteEnv?.BASE_URL ?? "/";
const publicAssetBaseUrl = appBaseUrl.endsWith("/") ? appBaseUrl : `${appBaseUrl}/`;
const API_BASE = (viteEnv?.VITE_MODEL_API_URL ?? "http://localhost:8000").replace(/\/$/, "");
const defaultInferenceParams: InferenceParams = { confidence: 0.25, iou: 0.7 };
const inferenceTimeoutMs = 90_000;
const maxUploadMiB = 10;
const maxUploadBytes = maxUploadMiB * 1024 * 1024;
const maxOriginalImageMegapixels = 24;
const maxOriginalImagePixels = maxOriginalImageMegapixels * 1_000_000;
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
  "latency_ms",
  "plant_count",
  "aruco_marker_count",
  "dataset_version",
  "training_map50_mask",
  "training_precision_mask",
  "training_recall_mask",
  "processed_image_width_px",
  "processed_image_height_px",
  "original_image_width_px",
  "original_image_height_px",
  "scale_calibrated",
  "scale_reason",
  "scale_dictionary",
  "scale_marker_size_mm",
  "scale_markers_detected",
  "scale_yolo_markers_detected",
  "scale_marker_ids",
  "scale_average_side_px",
  "scale_median_side_px",
  "scale_mm_per_pixel",
  "scale_pixels_per_mm",
  "plant_height_count",
  "plant_height_source_unit",
  "plant_height_calibrated",
  "plant_height_average_percent",
  "plant_height_max_percent",
  "plant_height_average_px",
  "plant_height_max_px",
  "plant_height_real_unit",
  "plant_height_average_mm",
  "plant_height_max_mm",
  "leaf_size_count",
  "leaf_size_source_unit",
  "leaf_size_calibrated",
  "leaf_average_width_percent",
  "leaf_average_height_percent",
  "leaf_average_area_percent",
  "leaf_average_width_px",
  "leaf_average_height_px",
  "leaf_average_area_px2",
  "leaf_size_real_unit",
  "leaf_average_width_mm",
  "leaf_average_height_mm",
  "leaf_average_area_mm2",
  "fruit_size_count",
  "fruit_size_source_unit",
  "fruit_size_calibrated",
  "fruit_average_width_percent",
  "fruit_average_height_percent",
  "fruit_average_area_percent",
  "fruit_average_width_px",
  "fruit_average_height_px",
  "fruit_average_area_px2",
  "fruit_size_real_unit",
  "fruit_average_width_mm",
  "fruit_average_height_mm",
  "fruit_average_area_mm2",
  "fruit_states_method",
  "fruit_state_green_count",
  "fruit_state_ripening_count",
  "fruit_state_ripe_count",
  "fruit_state_undefined_count",
  "fruit_states_counts",
  "detection_measurements_json",
  "measurements_json"
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

const initialImageSrc = "https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?auto=format&fit=crop&w=1400&q=82";

const classStyle: Record<ClassName, { label: string; pluralLabel: string; color: string; soft: string }> = {
  folha: { label: "Folha", pluralLabel: "Folhas", color: "#42f58d", soft: "rgba(66, 245, 141, .25)" },
  fruto: { label: "Fruto", pluralLabel: "Frutos", color: "#ff5c7a", soft: "rgba(255, 92, 122, .25)" },
  marcador_aruco: { label: "Marcador", pluralLabel: "Marcadores", color: "#69c8ff", soft: "rgba(105, 200, 255, .25)" },
  planta_inteira: { label: "Planta", pluralLabel: "Plantas", color: "#f2d66d", soft: "rgba(242, 214, 109, .24)" }
};
const displayClasses: ClassName[] = ["planta_inteira", "folha", "fruto", "marcador_aruco"];
const classFilterOptions: Array<{ value: ClassName | "todas"; label: string; icon: React.ReactNode }> = [
  { value: "todas", label: "Todas", icon: <Sparkles size={16} /> },
  { value: "planta_inteira", label: "Planta", icon: <Layers3 size={16} /> },
  { value: "folha", label: "Folhas", icon: <Leaf size={16} /> },
  { value: "fruto", label: "Frutos", icon: <CircleDot size={16} /> },
  { value: "marcador_aruco", label: "Marcador", icon: <ScanLine size={16} /> }
];

const imageAccept = "image/*,.heic,.heif";
const heicExtensions = [".heic", ".heif"];

class UploadLimitError extends Error {
  readonly status = 413;

  constructor(message: string) {
    super(`413 — ${message}`);
    this.name = "UploadLimitError";
  }
}

class ApiResponseError extends Error {
  constructor(
    readonly status: number,
    message: string
  ) {
    super(message);
    this.name = "ApiResponseError";
  }
}

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
    image.onerror = () => reject(new Error("Não foi possível ler as dimensões da imagem."));
    image.src = url;
  });

const assertUploadSize = (file: File) => {
  if (file.size <= maxUploadBytes) return;
  throw new UploadLimitError(`a imagem excede o limite de ${maxUploadMiB} MiB.`);
};

const assertOriginalImageDimensions = (width: number, height: number) => {
  const pixels = width * height;
  if (pixels <= maxOriginalImagePixels) return;

  const megapixels = (pixels / 1_000_000).toLocaleString("pt-BR", {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1
  });
  throw new UploadLimitError(
    `a imagem original tem ${megapixels} MP e excede o limite de ${maxOriginalImageMegapixels} MP.`
  );
};

const readImageDimensions = async (blob: Blob) => {
  let bitmap: ImageBitmap | null = null;
  let objectUrl: string | null = null;

  try {
    try {
      bitmap = await createImageBitmap(blob);
      return { width: bitmap.width, height: bitmap.height };
    } catch (error) {
      objectUrl = URL.createObjectURL(blob);
      const image = await loadImageElement(objectUrl);
      return { width: image.naturalWidth || image.width, height: image.naturalHeight || image.height };
    }
  } finally {
    bitmap?.close();
    if (objectUrl) URL.revokeObjectURL(objectUrl);
  }
};

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

    assertOriginalImageDimensions(source.width, source.height);
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
    if (error instanceof UploadLimitError) throw error;
    throw new Error("Não foi possível converter HEIC para JPG.");
  }
};

const prepareImageFileForUpload = async (file: File) => {
  assertUploadSize(file);

  if (isHeicImageFile(file)) {
    const convertedFile = await convertHeicToJpeg(file);
    assertUploadSize(convertedFile);
    return convertedFile;
  }

  const { width, height } = await readImageDimensions(file);
  assertOriginalImageDimensions(width, height);
  return file;
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

const fetchFreshJson = async <T,>(url: string) => {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Falha ao carregar ${url}`);
  return (await response.json()) as T;
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

const getEmptyClassCounts = () =>
  displayClasses.reduce(
    (acc, className) => {
      acc[className] = 0;
      return acc;
    },
    {} as Record<ClassName, number>
  );

const isFiniteNumber = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const isConfidenceScore = (value: unknown): value is number => isFiniteNumber(value) && value >= 0 && value <= 1;

const countDetectionsByClass = (items: Detection[]) =>
  items.reduce((acc, detection) => {
    acc[detection.className] += 1;
    return acc;
  }, getEmptyClassCounts());

const normalizeApiCounts = (apiCounts: Record<string, number> | undefined, items: Detection[]) => {
  const fallback = countDetectionsByClass(items);
  return displayClasses.reduce((counts, className) => {
    counts[className] = isFiniteNumber(apiCounts?.[className]) ? apiCounts[className] : fallback[className];
    return counts;
  }, getEmptyClassCounts());
};

const getConfidenceSummary = (items: Detection[]) => {
  const scores = items.map((detection) => detection.confidence).filter(isConfidenceScore);
  if (!scores.length) return { average: null, count: 0 };

  return {
    average: scores.reduce((total, score) => total + score, 0) / scores.length,
    count: scores.length
  };
};

const getAverageConfidence = (items: Detection[]) => getConfidenceSummary(items).average ?? 0;

const formatConfidencePercent = (value: number | null) =>
  value === null
    ? "—"
    : `${(value * 100).toLocaleString("pt-BR", {
        minimumFractionDigits: 1,
        maximumFractionDigits: 1
      })}%`;

const toKnownClassName = (value: string): ClassName | null => {
  if (value === "folha" || value === "fruto" || value === "marcador_aruco" || value === "planta_inteira") return value;
  return null;
};

const getModelDisplayName = (model: string | ApiModelInfo | null | undefined) => {
  if (!model) return "Plant.AI YOLOv8-seg";
  if (typeof model === "string") return model;
  return model.name ?? model.id ?? "Plant.AI YOLOv8-seg";
};

const getModelClassLabels = (model: ApiModelInfo | null | undefined) => {
  if (!model?.classes?.length) return [];

  return model.classes.map((item) => {
    const name = typeof item === "string" ? item : item.name;
    const knownName = toKnownClassName(name);
    return knownName ? classStyle[knownName].label : name;
  });
};

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
  counts,
  model,
  datasetVersion,
  trainingMetric,
  latencyMs,
  image,
  measurements
}: {
  file: File;
  detections: Detection[];
  counts: Record<ClassName, number>;
  model: string;
  datasetVersion?: string;
  trainingMetric?: TrainingMetric | null;
  latencyMs: number | null;
  image: AnalyzeResponse["image"];
  measurements?: InferenceMeasurements;
}): AnalysisHistoryItem => {
  const detectionMeasurements = detections.map(
    ({ id, className, confidence, area, lab, parentId, state, size }): DetectionMeasurementRecord => ({
      id,
      className,
      confidence,
      area,
      lab,
      parentId,
      state,
      size
    })
  );

  return {
    id: generateRandomId("ANL"),
    plantId: generateRandomId("PLT"),
    analyzedAt: new Date().toISOString(),
    imageName: file.name,
    imageSizeBytes: file.size,
    imageSize: formatFileSize(file.size),
    imageType: file.type || "desconhecido",
    plantCount: counts.planta_inteira,
    leafCount: counts.folha,
    fruitCount: counts.fruto,
    markerCount: counts.marcador_aruco,
    totalDetections: detections.length,
    averageConfidence: getAverageConfidence(detections),
    model,
    datasetVersion,
    trainingMap50Mask: trainingMetric?.map50,
    trainingPrecisionMask: trainingMetric?.precision,
    trainingRecallMask: trainingMetric?.recall,
    latencyMs,
    imageWidth: image.width,
    imageHeight: image.height,
    originalImageWidth: image.originalWidth,
    originalImageHeight: image.originalHeight,
    measurements,
    detectionMeasurements
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

const formatBooleanForReport = (value: boolean | undefined) =>
  typeof value === "boolean" ? String(value) : null;

const stringifyForReport = (value: unknown) => {
  if (value === undefined || value === null) return "";
  try {
    return JSON.stringify(value);
  } catch (error) {
    return "";
  }
};

const getFruitStateCount = (item: AnalysisHistoryItem, keys: string[]) => {
  const counts = item.measurements?.fruitStates?.counts;
  if (!counts) return null;
  const matchedValue = keys.find((key) => isFiniteNumber(counts[key]));
  return matchedValue ? counts[matchedValue] : null;
};

const reportColumnGetters: Record<string, (item: AnalysisHistoryItem) => string | number | null> = {
  plant_id: (item) => item.plantId,
  analysis_id: (item) => item.id,
  analyzed_at: (item) => item.analyzedAt,
  image_name: (item) => item.imageName,
  image_size_bytes: (item) => item.imageSizeBytes,
  image_size: (item) => item.imageSize,
  image_type: (item) => item.imageType,
  plant_count: (item) => item.plantCount ?? null,
  leaf_count: (item) => item.leafCount,
  fruit_count: (item) => item.fruitCount,
  aruco_marker_count: (item) => item.markerCount ?? null,
  total_detections: (item) => item.totalDetections,
  average_confidence: (item) =>
    isFiniteNumber(item.averageConfidence) ? Number(item.averageConfidence.toFixed(4)) : null,
  model: (item) => item.model ?? null,
  dataset_version: (item) => item.datasetVersion ?? null,
  training_map50_mask: (item) => item.trainingMap50Mask ?? null,
  training_precision_mask: (item) => item.trainingPrecisionMask ?? null,
  training_recall_mask: (item) => item.trainingRecallMask ?? null,
  latency_ms: (item) => item.latencyMs,
  processed_image_width_px: (item) => item.imageWidth ?? null,
  processed_image_height_px: (item) => item.imageHeight ?? null,
  original_image_width_px: (item) => item.originalImageWidth ?? null,
  original_image_height_px: (item) => item.originalImageHeight ?? null,
  scale_calibrated: (item) => formatBooleanForReport(item.measurements?.scale?.calibrated),
  scale_reason: (item) => item.measurements?.scale?.reason ?? null,
  scale_dictionary: (item) => item.measurements?.scale?.dictionary ?? null,
  scale_marker_size_mm: (item) => item.measurements?.scale?.configuredMarkerSizeMm ?? null,
  scale_markers_detected: (item) => item.measurements?.scale?.markersDetected ?? null,
  scale_yolo_markers_detected: (item) => item.measurements?.scale?.yoloMarkersDetected ?? null,
  scale_marker_ids: (item) => item.measurements?.scale?.markerIds?.join("|") ?? null,
  scale_average_side_px: (item) => item.measurements?.scale?.averageSidePixels ?? null,
  scale_median_side_px: (item) => item.measurements?.scale?.medianSidePixels ?? null,
  scale_mm_per_pixel: (item) => item.measurements?.scale?.mmPerPixel ?? null,
  scale_pixels_per_mm: (item) => item.measurements?.scale?.pixelsPerMm ?? null,
  plant_height_count: (item) => item.measurements?.plantHeight?.count ?? null,
  plant_height_source_unit: (item) => item.measurements?.plantHeight?.unit ?? null,
  plant_height_calibrated: (item) => formatBooleanForReport(item.measurements?.plantHeight?.calibrated),
  plant_height_average_percent: (item) => item.measurements?.plantHeight?.average ?? null,
  plant_height_max_percent: (item) => item.measurements?.plantHeight?.max ?? null,
  plant_height_average_px: (item) => item.measurements?.plantHeight?.pixel?.average ?? null,
  plant_height_max_px: (item) => item.measurements?.plantHeight?.pixel?.max ?? null,
  plant_height_real_unit: (item) => item.measurements?.plantHeight?.real?.unit ?? null,
  plant_height_average_mm: (item) => item.measurements?.plantHeight?.real?.average ?? null,
  plant_height_max_mm: (item) => item.measurements?.plantHeight?.real?.max ?? null,
  leaf_size_count: (item) => item.measurements?.leafSize?.count ?? null,
  leaf_size_source_unit: (item) => item.measurements?.leafSize?.unit ?? null,
  leaf_size_calibrated: (item) => formatBooleanForReport(item.measurements?.leafSize?.calibrated),
  leaf_average_width_percent: (item) => item.measurements?.leafSize?.averageWidth ?? null,
  leaf_average_height_percent: (item) => item.measurements?.leafSize?.averageHeight ?? null,
  leaf_average_area_percent: (item) => item.measurements?.leafSize?.averageArea ?? null,
  leaf_average_width_px: (item) => item.measurements?.leafSize?.pixel?.averageWidth ?? null,
  leaf_average_height_px: (item) => item.measurements?.leafSize?.pixel?.averageHeight ?? null,
  leaf_average_area_px2: (item) => item.measurements?.leafSize?.pixel?.averageArea ?? null,
  leaf_size_real_unit: (item) => item.measurements?.leafSize?.real?.unit ?? null,
  leaf_average_width_mm: (item) => item.measurements?.leafSize?.real?.averageWidth ?? null,
  leaf_average_height_mm: (item) => item.measurements?.leafSize?.real?.averageHeight ?? null,
  leaf_average_area_mm2: (item) => item.measurements?.leafSize?.real?.averageAreaMm2 ?? null,
  fruit_size_count: (item) => item.measurements?.fruitSize?.count ?? null,
  fruit_size_source_unit: (item) => item.measurements?.fruitSize?.unit ?? null,
  fruit_size_calibrated: (item) => formatBooleanForReport(item.measurements?.fruitSize?.calibrated),
  fruit_average_width_percent: (item) => item.measurements?.fruitSize?.averageWidth ?? null,
  fruit_average_height_percent: (item) => item.measurements?.fruitSize?.averageHeight ?? null,
  fruit_average_area_percent: (item) => item.measurements?.fruitSize?.averageArea ?? null,
  fruit_average_width_px: (item) => item.measurements?.fruitSize?.pixel?.averageWidth ?? null,
  fruit_average_height_px: (item) => item.measurements?.fruitSize?.pixel?.averageHeight ?? null,
  fruit_average_area_px2: (item) => item.measurements?.fruitSize?.pixel?.averageArea ?? null,
  fruit_size_real_unit: (item) => item.measurements?.fruitSize?.real?.unit ?? null,
  fruit_average_width_mm: (item) => item.measurements?.fruitSize?.real?.averageWidth ?? null,
  fruit_average_height_mm: (item) => item.measurements?.fruitSize?.real?.averageHeight ?? null,
  fruit_average_area_mm2: (item) => item.measurements?.fruitSize?.real?.averageAreaMm2 ?? null,
  fruit_states_method: (item) => item.measurements?.fruitStates?.method ?? null,
  fruit_state_green_count: (item) => getFruitStateCount(item, ["verde", "green"]),
  fruit_state_ripening_count: (item) => getFruitStateCount(item, ["em_maturacao", "maturando", "ripening"]),
  fruit_state_ripe_count: (item) => getFruitStateCount(item, ["maduro", "ripe"]),
  fruit_state_undefined_count: (item) => getFruitStateCount(item, ["indefinido", "undefined", "unknown"]),
  fruit_states_counts: (item) => stringifyForReport(item.measurements?.fruitStates?.counts),
  detection_measurements_json: (item) => stringifyForReport(item.detectionMeasurements),
  measurements_json: (item) => stringifyForReport(item.measurements)
};

const normalizeReportHeaderColumn = (column: string) => column.trim().replace(/^"|"$/g, "");
const getReportColumns = (header: string) => {
  const templateColumns = header.split(",").map(normalizeReportHeaderColumn).filter(Boolean);
  return [...templateColumns, ...analysisReportHeaders.filter((column) => !templateColumns.includes(column))];
};
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
    <col min="5" max="${analysisReportHeaders.length}" width="18" customWidth="1" />
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
  const columns = getReportColumns(await getAnalysisReportTemplateHeader());
  const header = columns.join(",");
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

const normalizeInferenceParam = (value: unknown, fallback: number) =>
  isConfidenceScore(value) ? value : fallback;

const analyzeImageRequest = async (form: FormData, params: InferenceParams) => {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), inferenceTimeoutMs);
  const search = new URLSearchParams({
    confidence: String(params.confidence),
    iou: String(params.iou),
    includeImages: "true"
  });

  try {
    return await fetch(`${API_BASE}/api/v1/inferences?${search}`, {
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
  if (error instanceof UploadLimitError) return errorMessage;
  if (error instanceof ApiResponseError && error.status === 413) {
    return `413 — A API recusou a imagem. O limite é ${maxUploadMiB} MiB e ${maxOriginalImageMegapixels} MP na imagem original.`;
  }
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
  const [analysisSource, setAnalysisSource] = useState<AnalysisSource>("visualizer");
  const [selectedClass, setSelectedClass] = useState<ClassName | "todas">("todas");
  const [detections, setDetections] = useState<Detection[]>(initialDetections);
  const [analysisCounts, setAnalysisCounts] = useState<Record<ClassName, number> | null>(null);
  const [inferenceMeasurements, setInferenceMeasurements] = useState<InferenceMeasurements | null>(null);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetric[]>([]);
  const [trainingFinalMetric, setTrainingFinalMetric] = useState<TrainingMetric | null>(null);
  const [artifacts, setArtifacts] = useState<TrainingArtifact[]>([]);
  const [originalImageSrc, setOriginalImageSrc] = useState(initialImageSrc);
  const [apiAnnotatedImageSrc, setApiAnnotatedImageSrc] = useState<string | null>(null);
  const [analysisModel, setAnalysisModel] = useState("");
  const [datasetVersion, setDatasetVersion] = useState<string | null>(null);
  const [modelClassLabels, setModelClassLabels] = useState<string[]>([]);
  const [modelTrainingProgress, setModelTrainingProgress] = useState<{ completed?: number; total?: number }>({});
  const [inferenceParams, setInferenceParams] = useState<InferenceParams>(defaultInferenceParams);
  const [appliedConfidenceThreshold, setAppliedConfidenceThreshold] = useState<number | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [apiStatus, setApiStatus] = useState<"loading" | "online" | "offline">("loading");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [maskFocus, setMaskFocus] = useState(58);
  const [analysisZoom, setAnalysisZoom] = useState(1);
  const [analysisPan, setAnalysisPan] = useState<Pan>({ x: 0, y: 0 });
  const [imageAspect, setImageAspect] = useState(1.55);
  const [isDraggingImage, setIsDraggingImage] = useState(false);
  const [isInstancesExpanded, setIsInstancesExpanded] = useState(false);
  const [isMeasurementsExpanded, setIsMeasurementsExpanded] = useState(false);
  const [isArtifactsExpanded, setIsArtifactsExpanded] = useState(false);
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisHistoryItem[]>(readAnalysisHistory);
  const [isExportingHistory, setIsExportingHistory] = useState(false);
  const [historyMessage, setHistoryMessage] = useState("");
  const beforeSize = useElementSize(beforeCardRef);
  const afterSize = useElementSize(afterCardRef);
  const [message, setMessage] = useState("Carregando metadados do modelo...");

  useEffect(() => {
    const loadApiData = async () => {
      const [modelResult, summaryResult, metricsResult, artifactsResult] = await Promise.allSettled([
        fetchFreshJson<ApiModelInfo>(`${API_BASE}/api/v1/models/current`),
        fetchFreshJson<ApiTrainingSummary>(`${API_BASE}/api/v1/training/summary`),
        fetchFreshJson<{ series: ApiTrainingMetric[]; source?: string }>(`${API_BASE}/api/v1/training/metrics`),
        fetchCachedJson<{ artifacts: TrainingArtifact[] }>("training-artifacts", `${API_BASE}/api/v1/training/artifacts`)
      ]);
      let hasApiData = false;
      let finalMetric: TrainingMetric | null = null;

      if (modelResult.status === "fulfilled") {
        const info = modelResult.value;
        setAnalysisModel(getModelDisplayName(info));
        setDatasetVersion(info.datasetVersion ?? null);
        setModelClassLabels(getModelClassLabels(info));
        setModelTrainingProgress({ completed: info.training?.completedEpochs, total: info.training?.epochs });
        setInferenceParams({
          confidence: normalizeInferenceParam(info.inference?.defaultConfidence, defaultInferenceParams.confidence),
          iou: normalizeInferenceParam(info.inference?.defaultIou, defaultInferenceParams.iou)
        });
        finalMetric = info.training?.finalMetrics ? toTrainingMetric(info.training.finalMetrics) : null;
        hasApiData = true;
      }

      if (summaryResult.status === "fulfilled") {
        setDatasetVersion((current) => summaryResult.value.datasetVersion ?? current);
        finalMetric = summaryResult.value.final ? toTrainingMetric(summaryResult.value.final) : finalMetric;
        hasApiData = true;
      }

      if (metricsResult.status === "fulfilled") {
        const metrics = metricsResult.value.series
          .filter((metric) => isFiniteNumber(metric.epoch))
          .sort((left, right) => left.epoch - right.epoch)
          .map(toTrainingMetric);
        setTrainingMetrics(metrics);
        finalMetric = finalMetric ?? metrics[metrics.length - 1] ?? null;
        hasApiData = true;
      }

      if (artifactsResult.status === "fulfilled") {
        setArtifacts(artifactsResult.value.artifacts);
      }

      setTrainingFinalMetric(finalMetric);
      if (hasApiData) {
        setApiStatus("online");
        setMessage(`API conectada em ${API_BASE}`);
      } else {
        setApiStatus("offline");
        setMessage("API do modelo indisponível. Os metadados não puderam ser atualizados.");
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

  const counts = useMemo(() => analysisCounts ?? countDetectionsByClass(detections), [analysisCounts, detections]);

  const confidenceSummary = getConfidenceSummary(detections);
  const totalArea = detections.reduce((total, detection) => total + detection.area, 0);
  const visibleDetections = isAnalyzing ? [] : filteredDetections;
  const hasApiAnnotatedImage = Boolean(apiAnnotatedImageSrc);
  const isApiImageSelected = analysisSource === "api" && hasApiAnnotatedImage;
  const analysisBaseImageSrc = isApiImageSelected ? apiAnnotatedImageSrc! : originalImageSrc;
  const shouldRenderLocalDetections = !isApiImageSelected;
  const lastMetric = trainingFinalMetric ?? trainingMetrics[trainingMetrics.length - 1] ?? null;
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
    setMessage(isHeicImageFile(file) ? "Validando e convertendo HEIC para JPG..." : "Validando imagem antes da inferência...");

    try {
      const uploadFile = await prepareImageFileForUpload(file);
      const localPreview = URL.createObjectURL(uploadFile);
      setAnalysisSource("visualizer");
      setApiAnnotatedImageSrc(null);
      setDetections([]);
      setAnalysisCounts(null);
      setInferenceMeasurements(null);
      setLatencyMs(null);
      setAppliedConfidenceThreshold(null);
      setOriginalImageSrc(localPreview);
      setMessage(`Enviando ${isHeicImageFile(file) ? "JPG convertido" : "imagem"} (${formatFileSize(uploadFile.size)}) para inferência...`);

      const form = new FormData();
      form.append("file", uploadFile, uploadFile.name);

      const response = await analyzeImageRequest(form, inferenceParams);

      if (!response.ok) {
        throw new ApiResponseError(response.status, await response.text());
      }

      const payload = (await response.json()) as AnalyzeResponse;
      const nextDetections = payload.detections.map(toDetection).filter(Boolean) as Detection[];
      const nextCounts = normalizeApiCounts(payload.counts, nextDetections);
      const inferenceDatasetVersion = typeof payload.model === "string" ? datasetVersion ?? undefined : payload.model.datasetVersion;
      const historyItem = createAnalysisHistoryItem({
        file,
        detections: nextDetections,
        counts: nextCounts,
        model: getModelDisplayName(payload.model),
        datasetVersion: inferenceDatasetVersion,
        trainingMetric: lastMetric,
        latencyMs: payload.latencyMs,
        image: payload.image,
        measurements: payload.measurements
      });

      const originalDataUrl = payload.image.originalDataUrl ?? localPreview;
      setOriginalImageSrc(originalDataUrl);
      setApiAnnotatedImageSrc(payload.image.annotatedDataUrl ?? null);
      if (payload.image.width && payload.image.height) setImageAspect(payload.image.width / payload.image.height);
      setDetections(nextDetections);
      setAnalysisCounts(nextCounts);
      setInferenceMeasurements(payload.measurements ?? null);
      setLatencyMs(payload.latencyMs);
      setAppliedConfidenceThreshold(
        normalizeInferenceParam(payload.thresholds?.confidence, inferenceParams.confidence)
      );
      setAnalysisModel(getModelDisplayName(payload.model));
      if (typeof payload.model !== "string") {
        const modelDatasetVersion = payload.model.datasetVersion;
        setDatasetVersion((current) => modelDatasetVersion ?? current);
        setModelClassLabels(getModelClassLabels(payload.model));
      }
      setApiStatus("online");
      addAnalysisHistoryItem(historyItem);
      setMessage(`Inferência concluída em ${(payload.latencyMs / 1000).toFixed(2)}s`);
    } catch (error) {
      if (error instanceof ApiResponseError) {
        setApiStatus("online");
      } else if (!(error instanceof UploadLimitError)) {
        setApiStatus("offline");
      }
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
          <h1>Análise visual de segmentação{datasetVersion ? ` ${datasetVersion}` : ""}</h1>
        </div>
        <div className={`live-pill ${apiStatus}`}>
          <span />
          {apiStatus === "online" ? "API conectada" : apiStatus === "loading" ? "Conectando API" : "API indisponível"}
        </div>
      </header>

      <section className="dashboard-grid">
        <div className="main-stack">
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
              <h2>Segmentação da planta</h2>
              <p className="api-message">{message}</p>
              <p className="upload-limits">Até 10 MiB • imagem original com até 24 MP</p>
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

          <div className={`image-workbench ${viewMode}`} style={{ "--image-aspect": imageAspect } as React.CSSProperties}>
            <div className="image-card before" ref={beforeCardRef}>
              <div className="image-surface" style={beforeSurface}>
                <img src={originalImageSrc} alt="Imagem original da planta" onLoad={updateImageAspect} />
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
                  <img
                    src={analysisBaseImageSrc}
                    alt={isApiImageSelected ? "Imagem analisada retornada pela API" : "Imagem com análise montada pelo visualizador"}
                    onLoad={updateImageAspect}
                  />
                  {shouldRenderLocalDetections && viewMode !== "boxes" && visibleDetections.length > 0 && (
                    <MaskFocusOverlay detections={visibleDetections} imageSrc={analysisBaseImageSrc} />
                  )}
                  <div className="scan-grid" />
                  <div className="scan-beam" />
                  {shouldRenderLocalDetections &&
                    visibleDetections.map((detection) => (
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
                {shouldRenderLocalDetections && (
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
                )}
              </div>
              <span>{isAnalyzing ? "Processando" : isApiImageSelected ? "Foto da API" : "Visualizador"}</span>
            </div>
          </div>

          <div className="control-strip">
            <Segmented
              ariaLabel="Fonte da análise"
              value={analysisSource}
              options={[
                { value: "visualizer", label: "Visualizador", icon: <Layers3 size={16} /> },
                {
                  value: "api",
                  label: "Foto da API",
                  icon: <Images size={16} />,
                  disabled: !hasApiAnnotatedImage,
                  title: hasApiAnnotatedImage ? undefined : "A API não retornou uma imagem anotada"
                }
              ]}
              onChange={setAnalysisSource}
            />
            <Segmented
              ariaLabel="Modo de visualização"
              value={viewMode}
              options={[
                { value: "compare", label: "Comparar", icon: <Eye size={16} /> },
                { value: "masks", label: "Máscaras", icon: <Layers3 size={16} /> },
                { value: "boxes", label: "Boxes", icon: <Boxes size={16} /> }
              ]}
              onChange={setViewMode}
            />
            <Segmented
              ariaLabel="Filtrar classes"
              value={selectedClass}
              options={classFilterOptions}
              onChange={setSelectedClass}
            />
          </div>
          </article>

          <InferenceMeasurementsPanel
            isExpanded={isMeasurementsExpanded}
            measurements={inferenceMeasurements}
            isAnalyzing={isAnalyzing}
            onToggle={() => setIsMeasurementsExpanded((current) => !current)}
          />

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
                  <div className="instance-list expandable-content">
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
                  <div className="expandable-content">
                    <div className="model-summary">
                      <strong>{analysisModel || "Modelo não informado pela API"}</strong>
                      {datasetVersion && <span>Dataset: {datasetVersion}</span>}
                      {isFiniteNumber(modelTrainingProgress.completed) && isFiniteNumber(modelTrainingProgress.total) && (
                        <span>Épocas: {modelTrainingProgress.completed}/{modelTrainingProgress.total}</span>
                      )}
                      <span>
                        Inferência: confiança {(inferenceParams.confidence * 100).toFixed(0)}%, IoU {(inferenceParams.iou * 100).toFixed(0)}%
                      </span>
                      <span>{latencyMs ? `Última inferência: ${(latencyMs / 1000).toFixed(2)}s` : "Aguardando upload"}</span>
                      {modelClassLabels.length > 0 && <span>Classes: {modelClassLabels.join(", ")}</span>}
                    </div>
                    <ArtifactGrid artifacts={artifacts} />
                  </div>
                </>
              )}
            </article>
          </section>

          <section className="data-panel training-panel-wide">
            <PanelTitle icon={<Activity size={18} />} title={`Treinamento${datasetVersion ? ` ${datasetVersion}` : ""}`} />
            {trainingMetrics.length ? (
              <LineChart data={trainingMetrics} />
            ) : (
              <div className="processing-state">
                {apiStatus === "loading" ? "Carregando métricas" : "Métricas indisponíveis"}
              </div>
            )}
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
        </div>

        <aside className="side-stack">
          <div className="metric-grid">
            <Metric label="Plantas" value={isAnalyzing ? "-" : counts.planta_inteira} color={classStyle.planta_inteira.color} />
            <Metric label="Folhas" value={isAnalyzing ? "-" : counts.folha} color={classStyle.folha.color} />
            <Metric label="Frutos" value={isAnalyzing ? "-" : counts.fruto} color={classStyle.fruto.color} />
            <Metric label="Marcadores" value={isAnalyzing ? "-" : counts.marcador_aruco} color={classStyle.marcador_aruco.color} />
            <Metric
              label="Confiança média"
              value={isAnalyzing ? "—" : formatConfidencePercent(confidenceSummary.average)}
              color="#69c8ff"
              detail={
                isAnalyzing
                  ? "Processando"
                  : confidenceSummary.count
                    ? `${confidenceSummary.count} detecções · corte ${formatConfidencePercent(
                        appliedConfidenceThreshold ?? inferenceParams.confidence
                      )}`
                    : "Nenhuma detecção"
              }
              title="Média aritmética dos escores de confiança das detecções retornadas pela API."
            />
            <Metric label="Área seg." value={isAnalyzing ? "-" : `${totalArea.toFixed(1)}%`} color="#f2d66d" />
          </div>

          <section className="data-panel">
            <PanelTitle icon={<BarChart3 size={18} />} title="Contagem por classe" />
            {isAnalyzing ? (
              <div className="processing-state">Em processamento</div>
            ) : (
              <BarChart
                data={displayClasses.map((className) => ({
                  label: classStyle[className].pluralLabel,
                  value: counts[className],
                  color: classStyle[className].color
                }))}
              />
            )}
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
  const latestItems = history.slice(0, 5);

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

const measurementNumberFormatter = new Intl.NumberFormat("pt-BR", {
  minimumFractionDigits: 0,
  maximumFractionDigits: 2
});
const preciseMeasurementNumberFormatter = new Intl.NumberFormat("pt-BR", {
  minimumFractionDigits: 0,
  maximumFractionDigits: 3
});

const formatMeasurementNumber = (value: number | undefined, precise = false) =>
  isFiniteNumber(value) ? (precise ? preciseMeasurementNumberFormatter : measurementNumberFormatter).format(value) : null;

const formatLinearMeasurement = (value: number | undefined, unit?: string) => {
  const normalizedValue = unit === "mm" ? value === undefined ? undefined : value / 10 : value;
  const formatted = formatMeasurementNumber(normalizedValue);
  if (!formatted) return null;
  if (unit === "image_percent") return `${formatted}%`;
  if (unit === "px") return `${formatted} px`;
  if (unit === "mm") return `${formatted} cm`;
  return unit ? `${formatted} ${unit}` : formatted;
};

const formatAreaMeasurement = (value: number | undefined, unit?: string) => {
  const normalizedValue = unit === "mm" || unit === "mm2" || unit === "mm²" ? value === undefined ? undefined : value / 100 : value;
  const formatted = formatMeasurementNumber(normalizedValue, true);
  if (!formatted) return null;
  if (unit === "image_percent") return `${formatted}% da imagem`;
  if (unit === "px") return `${formatted} px²`;
  if (unit === "mm" || unit === "mm2" || unit === "mm²") return `${formatted} cm²`;
  return unit ? `${formatted} ${unit}` : formatted;
};

const humanizeApiValue = (value: string) =>
  value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());

const fruitStateLabels: Record<string, string> = {
  verde: "Verde",
  green: "Verde",
  em_maturacao: "Em maturação",
  maturando: "Em maturação",
  ripening: "Em maturação",
  maduro: "Maduro",
  ripe: "Maduro",
  indefinido: "Indefinido",
  undefined: "Indefinido",
  unknown: "Indefinido"
};

const scaleReasonLabels: Record<string, string> = {
  no_aruco_marker_detected: "Nenhum marcador ArUco detectado.",
  missing_marker_size_mm: "Tamanho físico do marcador não configurado.",
  missing_marker_size: "Tamanho físico do marcador não configurado."
};

function InferenceMeasurementsPanel({
  isExpanded,
  measurements,
  isAnalyzing,
  onToggle
}: {
  isExpanded: boolean;
  measurements: InferenceMeasurements | null;
  isAnalyzing: boolean;
  onToggle: () => void;
}) {
  return (
    <section className={`data-panel measurements-panel ${isExpanded ? "is-expanded" : ""}`}>
      <div className="panel-title-row">
        <PanelTitle icon={<Ruler size={18} />} title="Medições da inferência" />
        <div className="panel-actions">
          <button
            aria-expanded={isExpanded}
            aria-label={isExpanded ? "Recolher medições da inferência" : "Expandir medições da inferência"}
            className="expand-button"
            onClick={onToggle}
            type="button"
          >
            <ChevronDown size={17} />
          </button>
        </div>
      </div>
      {isExpanded && (
        isAnalyzing ? (
          <div className="processing-state">Calculando medições</div>
        ) : measurements ? (
          <div className="measurement-list expandable-content">
            <ScaleMeasurementRow scale={measurements.scale} />
            <HeightMeasurementRow measurement={measurements.plantHeight} />
            <SizeMeasurementRow className="fruto" measurement={measurements.fruitSize} title="Frutos" />
            <SizeMeasurementRow className="folha" measurement={measurements.leafSize} title="Folhas" />
            <FruitStatesRow states={measurements.fruitStates} />
          </div>
        ) : (
          <p className="empty-state">As medições aparecerão após uma inferência.</p>
        )
      )}
    </section>
  );
}

function ScaleMeasurementRow({ scale }: { scale?: ScaleMeasurement }) {
  if (!scale) return null;
  const markerDetails = [
    scale.dictionary,
    isFiniteNumber(scale.configuredMarkerSizeMm) ? `${formatMeasurementNumber(scale.configuredMarkerSizeMm)} mm` : null,
    scale.markerIds?.length ? `IDs ${scale.markerIds.join(", ")}` : null
  ].filter(Boolean);

  return (
    <div className="measurement-row calibration-row">
      <div className="measurement-heading">
        <strong>Escala</strong>
        <span className={scale.calibrated ? "measurement-status calibrated" : "measurement-status"}>
          {scale.calibrated ? "Calibrada" : "Relativa"}
        </span>
      </div>
      {!scale.calibrated && <p>{scaleReasonLabels[scale.reason ?? ""] ?? "Escala física indisponível."}</p>}
      {scale.calibrated && isFiniteNumber(scale.mmPerPixel) && (
        <p>{formatMeasurementNumber(scale.mmPerPixel, true)} mm por pixel</p>
      )}
      {markerDetails.length > 0 && <small>{markerDetails.join(" · ")}</small>}
    </div>
  );
}

function HeightMeasurementRow({ measurement }: { measurement?: HeightMeasurement }) {
  if (!measurement) return null;
  const count = isFiniteNumber(measurement.count) ? measurement.count : null;

  if (count === 0) {
    return <EmptyMeasurementRow title="Altura das plantas" message="Nenhuma planta medida" />;
  }

  const physicalAverage = formatLinearMeasurement(measurement.real?.average, measurement.real?.unit);
  const physicalMax = formatLinearMeasurement(measurement.real?.max, measurement.real?.unit);
  const relativeAverage = formatLinearMeasurement(measurement.average, measurement.unit);
  const relativeMax = formatLinearMeasurement(measurement.max, measurement.unit);
  const pixelAverage = formatLinearMeasurement(measurement.pixel?.average, measurement.pixel?.unit ?? "px");
  const pixelMax = formatLinearMeasurement(measurement.pixel?.max, measurement.pixel?.unit ?? "px");

  return (
    <div className="measurement-row">
      <div className="measurement-heading">
        <strong>Altura das plantas</strong>
        {count !== null && <span>{count} medidas</span>}
      </div>
      {(physicalAverage || relativeAverage) && (
        <p>Média {physicalAverage ?? relativeAverage} · máxima {physicalMax ?? relativeMax ?? "não disponível"}</p>
      )}
      {physicalAverage && relativeAverage && <small>Relativa: {relativeAverage} · máxima {relativeMax}</small>}
      {pixelAverage && <small>Pixels: {pixelAverage} · máxima {pixelMax ?? "não disponível"}</small>}
    </div>
  );
}

function SizeMeasurementRow({
  className,
  measurement,
  title
}: {
  className: "folha" | "fruto";
  measurement?: SizeMeasurement;
  title: string;
}) {
  if (!measurement) return null;
  const count = isFiniteNumber(measurement.count) ? measurement.count : null;
  if (count === 0) return <EmptyMeasurementRow title={title} message={`Nenhum${className === "folha" ? "a" : ""} ${className} medid${className === "folha" ? "a" : "o"}`} />;

  const realWidth = formatLinearMeasurement(measurement.real?.averageWidth, measurement.real?.unit);
  const realHeight = formatLinearMeasurement(measurement.real?.averageHeight, measurement.real?.unit);
  const realArea = formatAreaMeasurement(measurement.real?.averageAreaMm2, "mm2");
  const relativeWidth = formatLinearMeasurement(measurement.averageWidth, measurement.unit);
  const relativeHeight = formatLinearMeasurement(measurement.averageHeight, measurement.unit);
  const relativeArea = formatAreaMeasurement(measurement.averageArea, measurement.unit);
  const pixelWidth = formatLinearMeasurement(measurement.pixel?.averageWidth, measurement.pixel?.unit ?? "px");
  const pixelHeight = formatLinearMeasurement(measurement.pixel?.averageHeight, measurement.pixel?.unit ?? "px");
  const pixelArea = formatAreaMeasurement(measurement.pixel?.averageArea, measurement.pixel?.unit ?? "px");

  return (
    <div className="measurement-row">
      <div className="measurement-heading">
        <strong>{title}</strong>
        {count !== null && <span>{count} medidas</span>}
      </div>
      {(realWidth || relativeWidth) && (
        <p>
          Média {realWidth ?? relativeWidth} × {realHeight ?? relativeHeight ?? "não disponível"}
          {(realArea ?? relativeArea) ? ` · área ${realArea ?? relativeArea}` : ""}
        </p>
      )}
      {realWidth && relativeWidth && <small>Relativa: {relativeWidth} × {relativeHeight} · área {relativeArea}</small>}
      {pixelWidth && <small>Pixels: {pixelWidth} × {pixelHeight} · área {pixelArea}</small>}
    </div>
  );
}

function EmptyMeasurementRow({ title, message }: { title: string; message: string }) {
  return (
    <div className="measurement-row">
      <div className="measurement-heading">
        <strong>{title}</strong>
      </div>
      <p>{message}</p>
    </div>
  );
}

function FruitStatesRow({ states }: { states?: InferenceMeasurements["fruitStates"] }) {
  const counts = states?.counts;
  if (!counts || !Object.keys(counts).length) return null;
  const entries = Object.entries(counts)
    .filter(([, count]) => isFiniteNumber(count))
    .sort((left, right) => right[1] - left[1]);
  if (!entries.length) return null;

  return (
    <div className="measurement-row">
      <div className="measurement-heading">
        <strong>Estado dos frutos</strong>
        {states?.method && <span>{states.method === "lab_color_heuristic" ? "Cor CIELAB" : humanizeApiValue(states.method)}</span>}
      </div>
      <div className="state-counts">
        {entries.map(([state, count]) => (
          <span key={state}>{fruitStateLabels[state] ?? humanizeApiValue(state)} <strong>{count}</strong></span>
        ))}
      </div>
    </div>
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
  const className = toKnownClassName(item.className);
  if (!className || !isConfidenceScore(item.confidence)) return null;
  const area = item.area ?? item.size?.imagePercent?.area ?? (item.bbox.w * item.bbox.h) / 100;

  return {
    id: item.id,
    className,
    confidence: item.confidence,
    bbox: item.bbox,
    mask: item.mask?.map((point) => `${point.x}% ${point.y}%`).join(", ") ?? "",
    area,
    lab: item.lab ?? { l: 0, a: 0, b: 0 },
    parentId: item.parentId,
    state: item.state,
    size: item.size
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
          <span>{formatConfidencePercent(detection.confidence)}</span>
        </div>
      )}
    </div>
  );
}

function MaskFocusOverlay({ detections, imageSrc }: { detections: Detection[]; imageSrc: string }) {
  const maskedDetections = detections.filter((detection) => detection.mask);

  return (
    <div className="mask-focus-layer" aria-hidden="true">
      <div className="mask-dim" />
      {maskedDetections.map((detection) => (
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

function Metric({
  label,
  value,
  color,
  detail,
  title
}: {
  label: string;
  value: string | number;
  color: string;
  detail?: string;
  title?: string;
}) {
  return (
    <div className="metric-card" style={{ "--accent": color } as React.CSSProperties} title={title}>
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
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
  if (!data.length) return null;
  const width = 420;
  const height = 170;
  const padding = 18;
  const divisor = Math.max(1, data.length - 1);
  const maxLoss = Math.max(1, ...data.map((item) => item.loss));
  const points = data
    .map((item, index) => {
      const x = data.length === 1 ? width / 2 : padding + (index / divisor) * (width - padding * 2);
      const y = height - padding - item.map50 * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");
  const lossPoints = data
    .map((item, index) => {
      const x = data.length === 1 ? width / 2 : padding + (index / divisor) * (width - padding * 2);
      const normalizedLoss = Math.min(item.loss / maxLoss, 1);
      const y = padding + normalizedLoss * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg className="line-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Curva de treinamento do modelo atual">
      <polyline points={lossPoints} fill="none" stroke="#ffb86b" strokeWidth="3" opacity=".75" />
      <polyline points={points} fill="none" stroke="#42f58d" strokeWidth="4" />
      {data.map((item, index) => {
        const x = data.length === 1 ? width / 2 : padding + (index / divisor) * (width - padding * 2);
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
  const relativeWidth = formatLinearMeasurement(detection.size?.imagePercent?.width, "image_percent");
  const relativeHeight = formatLinearMeasurement(detection.size?.imagePercent?.height, "image_percent");
  const relativeArea = formatAreaMeasurement(detection.size?.imagePercent?.area, "image_percent");
  const pixelWidth = formatLinearMeasurement(detection.size?.pixel?.width, detection.size?.pixel?.unit ?? "px");
  const pixelHeight = formatLinearMeasurement(detection.size?.pixel?.height, detection.size?.pixel?.unit ?? "px");
  const pixelArea = formatAreaMeasurement(detection.size?.pixel?.area, detection.size?.pixel?.unit ?? "px");
  const realWidth = formatLinearMeasurement(detection.size?.real?.width, detection.size?.real?.unit);
  const realHeight = formatLinearMeasurement(detection.size?.real?.height, detection.size?.real?.unit);
  const realArea = formatAreaMeasurement(detection.size?.real?.areaMm2, "mm2");
  const parentLabel = detection.parentId?.replace(/^planta_inteira-/, "Planta ");

  return (
    <div className="instance-row">
      <div className="chip" style={{ borderColor: theme.color }}>
        <span style={{ background: theme.color }} />
        {theme.label}
      </div>
      <strong>{formatConfidencePercent(detection.confidence)}</strong>
      <small className="instance-details">
        {realWidth && <span>Real: {realWidth} × {realHeight} · área {realArea}</span>}
        {relativeWidth && <span>Imagem: {relativeWidth} × {relativeHeight} · área {relativeArea}</span>}
        {pixelWidth && <span>Pixels: {pixelWidth} × {pixelHeight} · área {pixelArea}</span>}
        <span>CIELAB: L* {detection.lab.l} · a* {detection.lab.a} · b* {detection.lab.b}</span>
        {(detection.state || parentLabel) && (
          <span>
            {detection.state && `Estado: ${fruitStateLabels[detection.state] ?? humanizeApiValue(detection.state)}`}
            {detection.state && parentLabel ? " · " : ""}
            {parentLabel && `Associada a ${parentLabel}`}
          </span>
        )}
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
  ariaLabel,
  value,
  options,
  onChange
}: {
  ariaLabel: string;
  value: T;
  options: Array<{ value: T; label: string; icon: React.ReactNode; disabled?: boolean; title?: string }>;
  onChange: (value: T) => void;
}) {
  return (
    <div className="segmented" role="group" aria-label={ariaLabel}>
      {options.map((option) => (
        <button
          aria-pressed={option.value === value}
          className={option.value === value ? "active" : ""}
          disabled={option.disabled}
          key={option.value}
          onClick={() => onChange(option.value)}
          title={option.title}
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
