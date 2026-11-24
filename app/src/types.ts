export type DevicePref = "auto" | "cuda" | "cpu";
export type StageName = "downloading" | "analyzing" | "notes" | "layers" | "writing";

export interface Settings {
  customLevels: string;
  device: DevicePref;
  exportDir: string;
  engineVersionInstalled: string | null;
}

export interface YouTubeItem {
  id: string;
  url: string;
  title: string;
  artist: string;
  track_title: string;
  channel: string;
  duration_s: number | null;
  thumbnail_url: string | null;
}

export interface EngineInfo {
  version: string;
  cuda_available: boolean;
  gpu_name: string | null;
  active_device: string;
  detected_custom_levels: string | null;
  has_layer_models: boolean;
  missing_layer_models: string[];
  environments: string[];
  styles: string[];
}

export interface MapGenerationConfig {
  mode: "pack" | "custom";
  levels: string[] | null;
  nps: number | null;
  styles: string[];
  ornaments: number | null;
  lights: number;
  environment: string;
  seed: number;
  lightsRules: boolean;
}

export interface StartGenerationRequest {
  sourceType: "youtube" | "file" | "cached";
  sourcePathOrUrl: string;
  title: string;
  artist: string;
  coverPath: string | null;
  config: MapGenerationConfig;
  regenerateId?: string;
}

export interface LevelSummary {
  rank: string;
  target_nps: number;
  n_notes: number;
  nps: number;
  n_arcs: number;
  n_chains: number;
  /** Per-level decor counts (engine >= multi-level decor fix); absent from older engines. */
  n_walls?: number;
  n_bombs?: number;
}

export interface GenerateOutput {
  folder: string;
  title: string;
  artist: string;
  bpm: number;
  grid_steady: boolean;
  n_notes: number;
  nps: number;
  rank: string;
  levels: LevelSummary[];
  n_walls: number;
  n_bombs: number;
  n_lights: number;
  environment: string;
  warnings: string[];
  used_rule_lights: boolean;
  device_used: string;
}

export interface LibraryEntry {
  id: string;
  title: string;
  artist: string;
  folder: string;
  createdAt: string;
  bpm: number;
  levels: Array<{
    rank: string;
    nps: number;
    nNotes: number;
    nArcs: number;
    nChains: number;
    nWalls?: number;
    nBombs?: number;
  }>;
  nWalls: number;
  nBombs: number;
  nLights: number;
  environment: string;
  audioCachePath: string;
  coverCachePath: string | null;
  config: MapGenerationConfig;
}

export interface ProgressEvent {
  stage: StageName;
  pct: number;
}

export type QueueItemStatus =
  | "waiting"
  | "downloading"
  | "running"
  | "done"
  | "failed"
  | "cancelled";

export interface QueueEnqueueRequest extends StartGenerationRequest {
  thumbnailUrl?: string | null;
}

export interface QueueItem {
  id: string;
  status: QueueItemStatus;
  request: StartGenerationRequest;
  thumbnailUrl: string | null;
  enqueuedAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  progress: ProgressEvent | null;
  cancelling?: boolean;
  errorCode: string | null;
  result: GenerateOutput | null;
  entry: LibraryEntry | null;
}

export interface QueueProgressEvent {
  id: string;
  stage: StageName;
  pct: number;
}

export interface EnginePackProgress {
  stage: "downloading" | "verifying" | "extracting";
  pct: number;
  bytesDownloaded: number;
  totalBytes: number;
  partIndex: number;
  partCount: number;
}

export interface UpdateStatus {
  state:
    | "idle"
    | "checking"
    | "available"
    | "not-available"
    | "downloading"
    | "downloaded"
    | "error";
  version?: string;
  pct?: number;
}

export interface StartupState {
  needsEnginePack: boolean;
  engineVersion: string;
  appVersion: string;
  settings: Settings;
  info: EngineInfo | null;
}

export interface ViewerDifficultyRaw {
  characteristic: string;
  rank: string;
  difficultyRank: number;
  filename: string;
  njs: number;
  njsOffset: number;
  raw: Record<string, unknown>;
}

export interface ViewerMapPackage {
  folder: string;
  title: string;
  artist: string;
  mapper: string;
  bpm: number;
  songTimeOffset: number;
  environment: string;
  coverDataUrl: string | null;
  audioBytes: Uint8Array | null;
  difficulties: ViewerDifficultyRaw[];
}

export interface BsmApi {
  getStartupState(): Promise<StartupState>;
  installEnginePack(): Promise<void>;
  getInfo(): Promise<EngineInfo>;
  getSettings(): Promise<Settings>;
  saveSettings(partial: Partial<Settings>): Promise<Settings>;
  validateCustomLevels(folder: string): Promise<{ valid: boolean; resolvedPath: string | null }>;
  pickFolder(title?: string): Promise<string | null>;
  pickAudioFile(): Promise<string | null>;
  pickCoverImage(): Promise<{ path: string; dataUrl: string } | null>;
  openPath(targetPath: string): Promise<void>;
  openExternalUrl(url: string): Promise<void>;
  openLogs(): Promise<void>;
  readCoverDataUrl(filePath: string): Promise<string | null>;
  searchYouTube(query: string): Promise<YouTubeItem[]>;
  resolveYouTube(url: string): Promise<YouTubeItem>;
  getQueue(): Promise<QueueItem[]>;
  enqueueGeneration(req: QueueEnqueueRequest): Promise<QueueItem>;
  cancelQueueItem(id: string): Promise<QueueItem[]>;
  retryQueueItem(id: string): Promise<QueueItem[]>;
  moveQueueItem(id: string, direction: "up" | "down"): Promise<QueueItem[]>;
  clearFinishedQueue(): Promise<QueueItem[]>;
  /** @deprecated Removed in Task 5 once Wizard.tsx and StepProgressDone.tsx migrate to the queue API. */
  startGeneration(req: StartGenerationRequest): Promise<{ result: GenerateOutput; entry: LibraryEntry }>;
  /** @deprecated Removed in Task 5 once StepProgressDone.tsx is retired. */
  cancelActive(): Promise<void>;
  listLibrary(): Promise<LibraryEntry[]>;
  deleteLibraryEntry(id: string): Promise<LibraryEntry[]>;
  exportMapZip(folder: string, suggestedName?: string): Promise<string>;
  checkForUpdates(): Promise<void>;
  installUpdate(): Promise<void>;
  readMapForViewer(folder: string): Promise<ViewerMapPackage>;
  onQueueChanged(cb: (items: QueueItem[]) => void): () => void;
  onQueueProgress(cb: (ev: QueueProgressEvent) => void): () => void;
  onLibraryChanged(cb: (entries: LibraryEntry[]) => void): () => void;
  /** @deprecated Removed in Task 5 once Wizard.tsx migrates to onQueueProgress. */
  onProgress(cb: (ev: ProgressEvent) => void): () => void;
  onEnginePackProgress(cb: (ev: EnginePackProgress) => void): () => void;
  onUpdateStatus(cb: (ev: UpdateStatus) => void): () => void;
}

declare global {
  interface Window {
    bsm: BsmApi;
  }
}
