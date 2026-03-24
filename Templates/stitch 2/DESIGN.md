# Design System Document: Sovereign Intelligence Interface

## 1. Overview & Creative North Star
### The Creative North Star: "The Sentinel Archive"
This design system moves away from the "flat dashboard" trope to embrace a high-fidelity, cinematic aesthetic. It is defined by **The Sentinel Archive**—a philosophy where data is not just displayed, but "illuminated." We achieve an elite cybersecurity feel by treating the UI as a multi-layered glass terminal. By utilizing deep tonal shifts, glowing status indicators, and intentional atmospheric depth, we transform a complex fraud detection system into a sophisticated, authoritative experience.

To break the "template" look, we utilize **Asymmetric Information Density**. Important verdicts are given massive "Display" typography and "Glowing" containers, while secondary metadata is tucked into "Surface-Lowest" layers, creating a clear cognitive hierarchy that feels editorial and premium.

---

## 2. Colors
Our palette is built on the abyss of deep space—utilizing dark navy foundations to let purple-blue accents "pierce" the interface.

*   **Foundation:** The core background is `surface` (`#10131c`). All depth is built *up* from here.
*   **The "No-Line" Rule:** Do not use 1px solid borders to define layout sections. Instead, use a background shift. For instance, a sidebar should be `surface_container_low` against a `surface` main content area. Boundaries are felt through tonal transitions, not drawn with lines.
*   **Surface Hierarchy & Nesting:**
    *   **Level 0 (Background):** `surface` (`#10131c`)
    *   **Level 1 (Sectioning):** `surface_container_low` (`#181b25`)
    *   **Level 2 (Active Cards):** `surface_container` (`#1c1f29`)
    *   **Level 3 (Popovers/Modals):** `surface_container_highest` (`#32343f`)
*   **The "Glass & Gradient" Rule:** Primary actions and high-alert status boxes must utilize the `primary_container` (`#8083ff`) to `secondary_container` (`#571bc1`) gradient. This creates a "liquid light" effect that signifies high-tech precision.

---

## 3. Typography
We use **Inter** exclusively. Its geometric clarity provides the "Technical Manual" feel required for fraud detection.

*   **Display (lg/md):** Reserved for high-level fraud scores or "Verdict" states. Use `display-md` with `on_surface` color and -0.02em letter spacing for a "hacker-premium" look.
*   **Headlines (lg/md):** Used for primary card titles. These should be bold and authoritative.
*   **Body (md/sm):** For data labels and Aadhaar metadata. Use `on_surface_variant` (`#c7c4d7`) for secondary body text to reduce visual noise.
*   **Labels (md/sm):** All-caps, tracked-out (+5%) labels for technical data points (e.g., "TIMESTAMP", "DEVICE_ID").

---

## 4. Elevation & Depth
In a cybersecurity context, depth equals security. We do not use "shadows" in the traditional sense; we use **Luminescence**.

*   **The Layering Principle:** Stack `surface_container_lowest` cards on top of `surface_container` backgrounds. This creates a "carved out" effect rather than a "floating" effect, making the data feel embedded in the system.
*   **Ambient Glows:** For "Floating" elements (like Tooltips), use a shadow with a 40px blur, 0% offset, and `primary` color at 8% opacity. This mimics a glowing screen effect.
*   **The "Ghost Border" Fallback:** Per the original spec, cards use an `outline_variant` at 6% opacity. This is the *only* permissible line in the system. It should feel like a faint reflection on the edge of a glass pane.
*   **Glassmorphism:** For overlays, apply `backdrop-filter: blur(12px)` combined with a 40% opaque `surface_container`.

---

## 5. Components

### Cards & Data Containers
*   **Rule:** Forbid divider lines within cards. Use `0.75rem` (Spacing 3) of vertical padding to separate data points.
*   **Style:** 16px (`lg`) rounded corners. Background: `surface_container`. Border: 1px Ghost Border (6% White).

### Buttons
*   **Primary:** Gradient from `primary` to `secondary`. Text is `on_primary_fixed`. No border.
*   **Secondary:** Ghost style. `outline` border at 20% opacity. Text is `primary`.
*   **Tertiary:** Plain text with `primary` color, used for "Dismiss" or "View Logs."

### Rounded Square Badges
*   **Spec:** 26x26px. Radius: `sm` (4px).
*   **Usage:** Used for status icons (Checkmark, Warning, X).
*   **Colors:** `error_container` for Fraud, `primary_container` for Verified.

### Progress Bars (Fraud Probability)
*   **Track:** `surface_container_highest`.
*   **Indicator:** Status-dependent (Green: `#4ade80`, Amber: `#fbbf24`, Red: `#f87171`).
*   **Polish:** Add a `10px` outer glow (drop-shadow) to the indicator in its own color to make it look like a glowing LED.

### Glowing Verdict Boxes
*   **Design:** A container with `surface_bright` background and a `2px` inner-glow of the status color. This is where the final "FRAUD DETECTED" or "AUTHENTIC" text resides.

---

## 6. Do's and Don'ts

### Do
*   **Do** use `2.5rem` (Spacing 10) of breathing room between major data modules.
*   **Do** use "Surface Tones" to highlight different data types (e.g., biometric data vs. demographic data).
*   **Do** ensure all "Error" states use the `error` (`#ffb4ab`) token for high-contrast legibility against the dark background.

### Don't
*   **Don't** use pure black (#000). The `surface` (`#10131c`) is required to maintain the "Navy" depth.
*   **Don't** use standard 1px grey dividers. They break the "Glass Terminal" immersion. Use whitespace or tonal blocks.
*   **Don't** use bright, saturated backgrounds for large areas. Keep saturation for the "Glow" elements and buttons only.