import type { ReactNode } from "react";
import { AlertTriangle, CheckCircle2, Download, RefreshCw, Sparkles } from "lucide-react";
import { EN } from "../i18n/en";
import type { UpdateStatus } from "../types";
import { Button } from "./ui/primitives";

const U = EN.updateCard;

export function UpdateStatusCard({
  status,
  appVersion,
}: {
  status: UpdateStatus;
  appVersion: string;
}) {
  const check = () => window.bsm.checkForUpdates();
  let tone = "border-zinc-800/90 bg-zinc-900/60 text-zinc-400";
  let icon: ReactNode = <RefreshCw className="h-4 w-4" />;
  let title: string = U.idleTitle;
  let sub: string | null = U.idleSub(appVersion);
  let action: ReactNode = (
    <Button variant="secondary" size="sm" onClick={check}>
      {U.check}
    </Button>
  );
  let progress = false;

  switch (status.state) {
    case "checking":
      icon = <RefreshCw className="h-4 w-4 animate-spin" />;
      title = U.checkingTitle;
      sub = null;
      action = (
        <Button variant="secondary" size="sm" disabled>
          {U.check}
        </Button>
      );
      break;
    case "not-available":
      tone = "border-emerald-500/25 bg-emerald-500/5 text-emerald-400";
      icon = <CheckCircle2 className="h-4 w-4" />;
      title = U.upToDateTitle;
      sub = U.upToDateSub(appVersion);
      action = (
        <Button variant="secondary" size="sm" onClick={check}>
          {U.checkAgain}
        </Button>
      );
      break;
    case "available":
    case "downloading":
      tone = "border-blue-500/25 bg-blue-500/5 text-blue-400";
      icon = <Download className="h-4 w-4" />;
      title = U.foundTitle(status.version);
      sub = U.downloading(status.pct);
      action = null;
      progress = true;
      break;
    case "downloaded":
      tone = "border-emerald-500/50 bg-emerald-500/15 text-emerald-300";
      icon = <Sparkles className="h-4 w-4" />;
      title = U.readyTitle(status.version);
      sub = U.readySub;
      action = (
        <Button size="sm" onClick={() => window.bsm.installUpdate()}>
          {U.install}
        </Button>
      );
      break;
    case "error":
      tone = "border-amber-500/25 bg-amber-500/5 text-amber-400";
      icon = <AlertTriangle className="h-4 w-4" />;
      title = U.errorTitle;
      sub = U.errorSub;
      action = (
        <Button variant="secondary" size="sm" onClick={check}>
          {U.tryAgain}
        </Button>
      );
      break;
  }

  const unknownPct = status.pct === undefined;
  const pct = Math.max(0, Math.min(100, status.pct ?? 0));
  return (
    <div
      data-testid="update-card"
      data-state={status.state}
      className={`rounded-xl border px-3.5 py-3 ${tone}`}
    >
      <div className="flex items-center gap-3">
        <div className="shrink-0">{icon}</div>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium text-zinc-100">{title}</div>
          {sub && <div className="text-xs text-zinc-400">{sub}</div>}
        </div>
        {action && <div className="shrink-0">{action}</div>}
      </div>
      {progress && (
        <div className="mt-2.5 h-1 w-full overflow-hidden rounded-full bg-zinc-800">
          <div
            data-testid="update-progress"
            className={`h-full rounded-full bg-blue-500 ${unknownPct ? "animate-pulse opacity-40" : ""}`}
            style={{ width: unknownPct ? "100%" : `${pct}%` }}
          />
        </div>
      )}
    </div>
  );
}
