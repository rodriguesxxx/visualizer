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
  Eye,
  ImageUp,
  Images,
  Layers3,
  Leaf,
  Minus,
  Plus,
  ScanLine,
  SlidersVertical,
  Sparkles
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

const API_BASE = (
  (import.meta as unknown as { env?: { VITE_MODEL_API_URL?: string } }).env?.VITE_MODEL_API_URL ??
  "http://localhost:8000"
).replace(/\/$/, "");
const inferenceTimeoutMs = 90_000;
const maxConvertedImageSide = 1280;
const jpegQuality = 0.82;
const apiCacheTtlMs = 12 * 60 * 60 * 1000;
const apiCachePrefix = "plant-ai-api-cache";
const artifactImageCacheName = "plant-ai-artifact-images-v1";
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

  const counts = useMemo(
    () =>
      detections.reduce(
        (acc, detection) => {
          acc[detection.className] += 1;
          return acc;
        },
        { folha: 0, fruto: 0 }
      ),
    [detections]
  );

  const confidence = detections.length
    ? detections.reduce((total, detection) => total + detection.confidence, 0) / detections.length
    : 0;
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
      setImageSrc(payload.image.originalDataUrl);
      setDetections(payload.detections.map(toDetection).filter(Boolean) as Detection[]);
      setLatencyMs(payload.latencyMs);
      setAnalysisModel(payload.model);
      setApiStatus("online");
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
