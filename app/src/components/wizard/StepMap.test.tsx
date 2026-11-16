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
    expect(screen.getByText(/kosza/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /zaawansowane/i }));
    fireEvent.click(screen.getByLabelText(/światła regułowe/i));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ lightsRules: true }));

    fireEvent.click(screen.getByRole("button", { name: /generuj/i }));
    expect(onGenerate).toHaveBeenCalled();
  });
});
