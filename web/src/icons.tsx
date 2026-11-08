/** Inline SVG icon set — no icon dependency. */

import React from "react";

type P = { size?: number; className?: string };

const base = (size: number) => ({
  width: size, height: size, viewBox: "0 0 24 24",
  fill: "none", stroke: "currentColor", strokeWidth: 1.8,
  strokeLinecap: "round" as const, strokeLinejoin: "round" as const,
});

export const IconChat = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><path d="M21 11.5a8.4 8.4 0 0 1-9 8.4 9 9 0 0 1-3.8-.8L3 21l1.9-5.1A8.4 8.4 0 0 1 12 3.1a8.4 8.4 0 0 1 9 8.4Z" /></svg>
);
export const IconTarget = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="5" /><circle cx="12" cy="12" r="1.4" /></svg>
);
export const IconBrain = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><path d="M9.5 3.5A2.5 2.5 0 0 0 7 6v.4A2.6 2.6 0 0 0 5 9c0 .9.4 1.7 1 2.2A2.6 2.6 0 0 0 5 13.5c0 1.1.7 2 1.7 2.4V16a2.5 2.5 0 0 0 2.8 2.5" /><path d="M14.5 3.5A2.5 2.5 0 0 1 17 6v.4A2.6 2.6 0 0 1 19 9c0 .9-.4 1.7-1 2.2a2.6 2.6 0 0 1 1 2.3c0 1.1-.7 2-1.7 2.4V16a2.5 2.5 0 0 1-2.8 2.5" /><path d="M12 3v18" /></svg>
);
export const IconSpark = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M18.4 5.6l-2.8 2.8M8.4 15.6l-2.8 2.8" /></svg>
);
export const IconShield = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><path d="M12 3l7 3v6c0 4.4-3 8.2-7 9-4-.8-7-4.6-7-9V6l7-3Z" /><path d="m9 12 2 2 4-4" /></svg>
);
export const IconSettings = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-2.9 1.2 2 2 0 1 1-4 0 1.7 1.7 0 0 0-2.9-1.2l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1A1.7 1.7 0 0 0 3 15a2 2 0 1 1 0-4 1.7 1.7 0 0 0 1.5-2.9l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1A1.7 1.7 0 0 0 10 4.6a2 2 0 1 1 4 0 1.7 1.7 0 0 0 2.9 1.2l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1A1.7 1.7 0 0 0 21 11a2 2 0 1 1 0 4Z" /></svg>
);
export const IconSend = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><path d="M4 12l16-8-6 8 6 8-16-8Z" /></svg>
);
export const IconStop = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none" /></svg>
);
export const IconPlus = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><path d="M12 5v14M5 12h14" /></svg>
);
export const IconTrash = ({ size = 16, className }: P) => (
  <svg {...base(size)} className={className}><path d="M4 7h16M9 7V5h6v2M6 7l1 13h10l1-13" /></svg>
);
export const IconMenu = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><path d="M4 6h16M4 12h16M4 18h16" /></svg>
);
export const IconSun = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4" /></svg>
);
export const IconMoon = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5Z" /></svg>
);
export const IconSearch = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></svg>
);
export const IconCode = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><path d="m9 18-6-6 6-6M15 6l6 6-6 6" /></svg>
);
export const IconGlobe = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3a15 15 0 0 1 0 18 15 15 0 0 1 0-18Z" /></svg>
);
export const IconCalc = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><rect x="4" y="3" width="16" height="18" rx="2" /><path d="M8 7h8M8 11h2M12 11h2M16 11h.01M8 15h2M12 15h2M16 15h.01" /></svg>
);
export const IconCheck = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><path d="m5 13 4 4L19 7" /></svg>
);
export const IconX = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><path d="M6 6l12 12M18 6 6 18" /></svg>
);
export const IconChevron = ({ size = 16, className }: P) => (
  <svg {...base(size)} className={className}><path d="m9 6 6 6-6 6" /></svg>
);
export const IconLayers = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 13 9 5 9-5" /></svg>
);
export const IconUsers = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><circle cx="9" cy="8" r="3.2" /><path d="M3 20a6 6 0 0 1 12 0" /><path d="M16 5.5a3 3 0 0 1 0 5.6M18 20a6 6 0 0 0-2-4.5" /></svg>
);
export const IconBolt = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z" /></svg>
);
export const IconClock = ({ size = 16, className }: P) => (
  <svg {...base(size)} className={className}><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></svg>
);
export const IconDatabase = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><ellipse cx="12" cy="6" rx="8" ry="3" /><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6" /><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" /></svg>
);
export const IconChart = ({ size = 18, className }: P) => (
  <svg {...base(size)} className={className}><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" /></svg>
);
export const IconCopy = ({ size = 15, className }: P) => (
  <svg {...base(size)} className={className}><rect x="9" y="9" width="12" height="12" rx="2" /><path d="M5 15V5a2 2 0 0 1 2-2h10" /></svg>
);
