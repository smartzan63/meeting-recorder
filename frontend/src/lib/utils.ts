import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Backend stores timestamps in UTC. Some are naive ISO (no offset), which JS
// would otherwise parse as local time. Append 'Z' when no timezone is present
// so the instant is read as UTC, then render in the viewer's local timezone,
// 24-hour format (no AM/PM).
export function formatLocalDateTime(
  isoString: string,
  opts: Intl.DateTimeFormatOptions = { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' },
): string {
  if (!isoString) return ''
  const hasTz = /(?:Z|[+-]\d{2}:?\d{2})$/.test(isoString)
  const normalized = hasTz ? isoString : `${isoString}Z`
  const d = new Date(normalized)
  if (isNaN(d.getTime())) return isoString
  // hourCycle 'h23' keeps midnight as 00:00 (some locales render it 24:00 with
  // only hour12:false).
  return d.toLocaleString(undefined, { ...opts, hour12: false, hourCycle: 'h23' })
}
