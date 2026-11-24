import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PACK_DEFAULT_CONFIG, StepMap } from "./StepMap";

describe("StepMap", () => {
  it("defaults to Pakiet, supports Własne controls and shows saved seed on regenerate", () => {
    const onChange = vi.fn();
    const onGenerate = vi.fn();
    render(
      <StepMap
        config={{ ...PACK_DEFAULT_CONFIG, mode: "custom", seed: 12345 }}
        isRegenerate
        onChange={onChange}
        onBack={() => {}}
        onGenerate={onGenerate}
      />,
    );
    expect(screen.getByDisplayValue("12345")).toBeInTheDocument();
    expect(screen.getByText(/recycle bin/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /advanced/i }));
    fireEvent.click(screen.getByLabelText(/rule-based lights/i));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ lightsRules: true }));

    fireEvent.click(screen.getByRole("button", { name: /add to queue|queue regeneration/i }));
    expect(onGenerate).toHaveBeenCalled();
  });

  it("shows Add to Queue when not regenerating", () => {
    render(
      <StepMap
        config={PACK_DEFAULT_CONFIG}
        isRegenerate={false}
        onChange={vi.fn()}
        onBack={vi.fn()}
        onGenerate={vi.fn()}
      />,
    );
    expect(screen.getByRole("button", { name: /add to queue/i })).toBeInTheDocument();
  });
});
