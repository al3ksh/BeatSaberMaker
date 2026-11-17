import { Settings as SettingsIcon } from "lucide-react";
import { EN } from "../i18n/en";
import type { UpdateStatus } from "../types";
import { Button } from "./ui/primitives";

export function SettingsButton({
  updateState,
  onClick,
}: {
  updateState: UpdateStatus["state"];
  onClick: () => void;
}) {
  const busy = updateState === "available" || updateState === "downloading";
  const ready = updateState === "downloaded";
  const label = busy
    ? EN.nav.settingsUpdateDownloading
    : ready
      ? EN.nav.settingsUpdateReady
      : EN.nav.settings;
  return (
    <Button
      variant="secondary"
      size="sm"
      onClick={onClick}
      aria-label={label}
      title={label}
      className="relative rounded-xl px-2.5 h-8 w-8 p-0"
    >
      <SettingsIcon className="h-4 w-4" />
      {(busy || ready) && (
        <span
          data-testid="settings-update-dot"
          aria-hidden="true"
          className={`absolute -top-0.5 -right-0.5 h-2.5 w-2.5 rounded-full ring-2 ring-zinc-950 ${
            busy ? "bg-blue-500 animate-pulse" : "bg-emerald-500"
          }`}
        />
      )}
    </Button>
  );
}
