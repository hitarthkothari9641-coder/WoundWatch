import { Platform } from 'react-native';
import Constants from 'expo-constants';

// On the web build the page and API share one origin (the Cloudflare Worker), so use relative URLs.
export const API = Platform.OS === 'web' ? '' : (process.env.EXPO_PUBLIC_API_URL || (Constants.expoConfig?.extra as any)?.apiUrl || '').replace(/\/$/, '');
let csrf = '';

// Session is an HttpOnly cookie held by the native cookie jar; only the CSRF token lives in memory.
async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { ...(init.headers as any) };
  if (csrf && init.method && init.method !== 'GET') headers['X-CSRF-Token'] = csrf;
  const res = await fetch(API + path, { ...init, headers, credentials: 'include' });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw Object.assign(new Error(typeof body.detail === 'string' ? body.detail : `Request failed (${res.status})`), { status: res.status });
  return body as T;
}

export type Scan = { id: string; scan_date: string; wound_location: string; patient_id: string;
  result: { status: string; probability: number | null; risk_band: string; reasons: string[] }; review_status: string };
export type Guidance = { summary: string; observations: string[]; check: string[]; disclaimer: string };
export type Patient = { id: string; name: string };
export type Doctor = { id: string; name: string; shared: boolean };

export async function login(email: string, password: string) {
  const r = await call<{ csrf_token: string; user: { name: string; role: string } }>('/api/auth/login', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email, password, account_type: 'patient' }) });
  csrf = r.csrf_token;
  return r.user;
}
export const logout = () => call('/api/auth/logout', { method: 'POST' }).finally(() => { csrf = ''; });
export const patients = () => call<Patient[]>('/api/patients');
export const scans = () => call<Scan[]>('/api/scans');
export const careTeam = () => call<Doctor[]>('/api/care-team');
export const setCareTeam = (ids: string[]) => call<Doctor[]>('/api/care-team', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ clinician_ids: ids }) });
export const guidance = (id: string) => call<Guidance>(`/api/scans/${id}/guidance`);

export async function upload(p: { uri: string; mime?: string; patientId: string; location: string; surgeryDate: string }) {
  const today = new Date().toISOString().slice(0, 10);
  const form = new FormData();
  if (Platform.OS === 'web') form.append('file', await (await fetch(p.uri)).blob(), 'wound.jpg');
  else form.append('file', { uri: p.uri, name: 'wound.jpg', type: p.mime || 'image/jpeg' } as any);
  form.append('patient_id', p.patientId); form.append('scan_date', today);
  form.append('wound_location', p.location); form.append('surgery_date', p.surgeryDate);
  return call<Scan>('/api/scans', { method: 'POST', body: form });
}

// Image views do not reliably share the cookie jar on Android, so load overlays through fetch as data URIs.
export async function overlay(id: string, kind: 'colour' | 'gradcam'): Promise<string> {
  const res = await fetch(`${API}/api/scans/${id}/visualization/${kind}`, { credentials: 'include' });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || 'Picture unavailable');
  const blob = await res.blob();
  return new Promise((ok, no) => { const r = new FileReader(); r.onload = () => ok(String(r.result)); r.onerror = no; r.readAsDataURL(blob); });
}
