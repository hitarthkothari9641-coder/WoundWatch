import { useEffect, useState } from 'react';
import { ActivityIndicator, Image, Pressable, SafeAreaView, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import * as ImagePicker from 'expo-image-picker';
import { useFonts, Outfit_800ExtraBold } from '@expo-google-fonts/outfit';
import { PlusJakartaSans_400Regular, PlusJakartaSans_600SemiBold } from '@expo-google-fonts/plus-jakarta-sans';
import { SpaceMono_400Regular } from '@expo-google-fonts/space-mono';
import * as Speech from 'expo-speech';
import * as api from './src/api';
import assist from './src/assist.json';
import { c, f } from './src/theme';

type Screen = { name: 'login' } | { name: 'help'; from: Screen } | { name: 'home' } | { name: 'scan' } | { name: 'result'; scan: api.Scan };
const BAND: Record<string, [string, string, string]> = {
  low: ['Low', c.okBg, '#0A5F63'], elevated: ['Elevated', c.badBg, '#A31B2A'], defer: ['Needs review', c.warnBg, '#8A5A00'], unknown: ['Retake photo', c.warnBg, '#8A5A00'] };

const Btn = ({ label, onPress, kind = 'primary', busy }: { label: string; onPress: () => void; kind?: 'primary' | 'ghost'; busy?: boolean }) => (
  <Pressable accessibilityRole="button" onPress={onPress} disabled={busy} style={[s.btn, kind === 'ghost' && s.btnGhost, busy && { opacity: 0.6 }]}>
    {busy ? <ActivityIndicator color={kind === 'ghost' ? c.teal : '#fff'} /> : <Text style={[s.btnText, kind === 'ghost' && { color: c.teal }]}>{label}</Text>}
  </Pressable>);
const Badge = ({ band }: { band: string }) => { const [t, bg, fg] = BAND[band] ?? BAND.unknown; return <View style={[s.badge, { backgroundColor: bg }]}><Text style={{ color: fg, fontFamily: f.bodyBold, fontSize: 13 }}>{t}</Text></View>; };

function Login({ onDone }: { onDone: () => void }) {
  const [email, setEmail] = useState(''); const [pw, setPw] = useState(''); const [err, setErr] = useState(''); const [busy, setBusy] = useState(false);
  const go = async () => { setBusy(true); setErr(''); try { await api.login(email.trim(), pw); onDone(); } catch (e: any) { setErr(e.message); } setBusy(false); };
  return (<ScrollView contentContainerStyle={s.pad} keyboardShouldPersistTaps="handled">
    <Image source={require('./assets/logo-horizontal.png')} style={{ width: 260, height: 73, resizeMode: 'contain', marginVertical: 24 }} accessibilityLabel="Wound Trace" />
    <Text style={s.h1}>A little care. A clearer picture.</Text>
    <Text style={s.p}>Sign in to photograph your wound and see how it is healing.</Text>
    <TextInput style={s.input} placeholder="Email" autoCapitalize="none" keyboardType="email-address" value={email} onChangeText={setEmail} />
    <TextInput style={s.input} placeholder="Password" secureTextEntry value={pw} onChangeText={setPw} />
    {!!err && <Text style={s.err} accessibilityRole="alert">{err}</Text>}
    <Btn label="Sign in" onPress={go} busy={busy} />
    <Text style={s.fine}>Research prototype. Not a diagnosis. Does not replace clinical care.</Text>
  </ScrollView>);
}

function Home({ go, signOut }: { go: (s: Screen) => void; signOut: () => void }) {
  const [list, setList] = useState<api.Scan[] | null>(null); const [err, setErr] = useState('');
  const [docs, setDocs] = useState<api.Doctor[]>([]); const [saved, setSaved] = useState('');
  useEffect(() => { api.careTeam().then(setDocs).catch(() => {}); }, []);
  const saveDocs = async () => { try { setDocs(await api.setCareTeam(docs.filter(d => d.shared).map(d => d.id))); setSaved('Saved. Your doctor can now see your record.'); } catch (e: any) { setSaved(e.message); } };
  useEffect(() => { api.scans().then(r => setList([...r].sort((a, b) => b.scan_date.localeCompare(a.scan_date)))).catch(e => setErr(e.message)); }, []);
  return (<ScrollView contentContainerStyle={s.pad}>
    <View style={s.row}><Image source={require('./assets/emblem.png')} style={{ width: 44, height: 44 }} /><Text style={s.h1}>My wound journey</Text></View>
    <Btn label="Scan my wound" onPress={() => go({ name: 'scan' })} />
    {err ? <Text style={s.err}>{err}</Text> : list === null ? <ActivityIndicator color={c.teal} style={{ margin: 24 }} /> : list.length === 0 ? <Text style={s.p}>No scans yet. Take your first photograph to start your timeline.</Text> :
      list.map(sc => (<Pressable key={sc.id} style={s.card} onPress={() => go({ name: 'result', scan: sc })} accessibilityRole="button">
        <View style={s.row}><Text style={s.h3}>{sc.wound_location}</Text><Badge band={sc.result.risk_band} /></View>
        <Text style={[s.p, { fontFamily: f.mono }]}>{sc.scan_date}</Text></Pressable>))}
    <View style={s.card}><Text style={s.h3}>My doctor</Text>
      <Text style={s.p}>Choose who can see your photographs and details.</Text>
      {docs.length === 0 ? <Text style={s.p}>No doctors have joined yet.</Text> : docs.map(d => (
        <Pressable key={d.id} accessibilityRole="checkbox" accessibilityState={{ checked: d.shared }} style={[s.chip, d.shared && s.chipOn, { alignItems: 'flex-start' }]}
          onPress={() => setDocs(docs.map(x => x.id === d.id ? { ...x, shared: !x.shared } : x))}><Text style={[s.chipText, d.shared && { color: '#fff' }]}>{d.shared ? '\u2713 ' : ''}{d.name}</Text></Pressable>))}
      {docs.length > 0 && <Btn label="Save" onPress={saveDocs} />}{!!saved && <Text style={s.fine}>{saved}</Text>}</View>
    <Btn kind="ghost" label="Sign out" onPress={signOut} />
  </ScrollView>);
}

function Scan({ done, back }: { done: (sc: api.Scan) => void; back: () => void }) {
  const [loc, setLoc] = useState(''); const [surg, setSurg] = useState(''); const [uri, setUri] = useState(''); const [mime, setMime] = useState<string>();
  const [busy, setBusy] = useState(false); const [err, setErr] = useState('');
  const pick = async (camera: boolean) => {
    const perm = camera ? await ImagePicker.requestCameraPermissionsAsync() : await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) { setErr('Permission is needed to continue.'); return; }
    const r = await (camera ? ImagePicker.launchCameraAsync : ImagePicker.launchImageLibraryAsync)({ mediaTypes: ['images'], quality: 0.9 });
    if (!r.canceled) { setUri(r.assets[0].uri); setMime(r.assets[0].mimeType ?? undefined); setErr(''); }
  };
  const submit = async () => {
    if (!uri || !loc.trim() || !/^\d{4}-\d{2}-\d{2}$/.test(surg)) { setErr('Add a photo, the wound location, and the surgery date (YYYY-MM-DD).'); return; }
    setBusy(true); setErr('');
    try { const [p] = await api.patients(); const sc = await api.upload({ uri, mime, patientId: p.id, location: loc.trim(), surgeryDate: surg }); done(sc); }
    catch (e: any) { setErr(e.message); } setBusy(false);
  };
  return (<ScrollView contentContainerStyle={s.pad} keyboardShouldPersistTaps="handled">
    <Text style={s.h1}>New photograph</Text>
    <Text style={s.p}>Even light, whole wound in frame, hold still until it is sharp.</Text>
    {!!uri && <Image source={{ uri }} style={s.photo} />}
    <Btn label={uri ? 'Retake photo' : 'Take photo'} onPress={() => pick(true)} />
    <Btn kind="ghost" label="Choose from library" onPress={() => pick(false)} />
    <TextInput style={s.input} placeholder="Wound location (e.g. left knee)" value={loc} onChangeText={setLoc} />
    <TextInput style={s.input} placeholder="Surgery date YYYY-MM-DD" value={surg} onChangeText={setSurg} keyboardType="numbers-and-punctuation" />
    {!!err && <Text style={s.err} accessibilityRole="alert">{err}</Text>}
    <Btn label="Analyse" onPress={submit} busy={busy} />
    <Btn kind="ghost" label="Back" onPress={back} />
  </ScrollView>);
}

function Result({ scan, back }: { scan: api.Scan; back: () => void }) {
  const [g, setG] = useState<api.Guidance | null>(null); const [img, setImg] = useState<Record<string, string>>({}); const [err, setErr] = useState('');
  useEffect(() => {
    api.guidance(scan.id).then(setG).catch(e => setErr(e.message));
    (['gradcam', 'colour'] as const).forEach(k => api.overlay(scan.id, k).then(u => setImg(p => ({ ...p, [k]: u }))).catch(() => {}));
  }, [scan.id]);
  const p = scan.result.probability;
  return (<ScrollView contentContainerStyle={s.pad}>
    <View style={s.row}><Text style={s.h1}>{scan.wound_location}</Text><Badge band={scan.result.risk_band} /></View>
    {p != null && <Text style={[s.big, { fontFamily: f.mono }]}>{Math.round(p * 100)}%</Text>}
    {p != null && <Text style={s.p}>Estimated Elevated probability (screening signal for your care team).</Text>}
    {scan.result.status === 'retake' && <Text style={s.err}>{scan.result.reasons.join('. ')}. Please retake the photo.</Text>}
    {err ? <Text style={s.err}>{err}</Text> : !g ? <ActivityIndicator color={c.teal} /> : (<View style={[s.card, { borderLeftColor: c.teal, borderLeftWidth: 4 }]}>
      <Text style={s.h3}>{g.summary}</Text>
      {g.observations.map((o, i) => <Text key={i} style={s.li}>• {o}</Text>)}
      <Text style={[s.h3, { marginTop: 12 }]}>What to check</Text>
      {g.check.map((o, i) => <Text key={i} style={s.li}>• {o}</Text>)}
      <Text style={s.fine}>{g.disclaimer}</Text></View>)}
    {img.gradcam && <><Text style={s.h3}>Where the model looked</Text><Image source={{ uri: img.gradcam }} style={s.photo} accessibilityLabel="Grad-CAM attention map" /></>}
    {img.colour && <><Text style={s.h3}>Measured redness-green map</Text><Image source={{ uri: img.colour }} style={s.photo} accessibilityLabel="Colour heatmap" /></>}
    <Btn kind="ghost" label="Back" onPress={back} />
  </ScrollView>);
}

function Help({ back }: { back: () => void }) {
  const [lang, setLang] = useState<'en' | 'hi' | 'gu'>('en');
  const ui = assist.ui[lang]; const code = assist.langs[lang][1];
  return (<ScrollView contentContainerStyle={s.pad}>
    <View style={s.row}><Text style={s.h1}>{ui.title}</Text></View>
    <View style={[s.row, { justifyContent: 'flex-start', flexWrap: 'wrap' }]} accessibilityLabel={ui.lang}>
      {(Object.keys(assist.langs) as ('en' | 'hi' | 'gu')[]).map(k => (
        <Pressable key={k} accessibilityRole="button" accessibilityState={{ selected: k === lang }} onPress={() => { Speech.stop(); setLang(k); }} style={[s.chip, k === lang && s.chipOn]}>
          <Text style={[s.chipText, k === lang && { color: '#fff' }]}>{assist.langs[k][0]}</Text></Pressable>))}
    </View>
    {assist.sections.filter(sec => !(sec as { web?: boolean }).web).map(sec => { const steps = sec.steps[lang]; return (
      <View key={sec.id} style={s.card}>
        <Text style={s.h3}>{sec.title[lang]}</Text>
        {steps.map((t, i) => <Text key={i} style={s.li}>{i + 1}. {t}</Text>)}
        <View style={[s.row, { justifyContent: 'flex-start' }]}>
          <Pressable accessibilityRole="button" style={[s.chip, s.chipOn]} onPress={() => Speech.speak(sec.title[lang] + '. ' + steps.join('. '), { language: code, rate: 0.9 })}><Text style={[s.chipText, { color: '#fff' }]}>{ui.read}</Text></Pressable>
          <Pressable accessibilityRole="button" style={s.chip} onPress={() => Speech.stop()}><Text style={s.chipText}>{ui.stop}</Text></Pressable>
        </View>
      </View>); })}
    <Btn kind="ghost" label={ui.close} onPress={() => { Speech.stop(); back(); }} />
  </ScrollView>);
}

export default function App() {
  const [fontsOk] = useFonts({ Outfit_800ExtraBold, PlusJakartaSans_400Regular, PlusJakartaSans_600SemiBold, SpaceMono_400Regular });
  const [screen, setScreen] = useState<Screen>({ name: 'login' });
  if (!fontsOk) return <View style={{ flex: 1, backgroundColor: c.bg }} />;
  const home = () => setScreen({ name: 'home' });
  return (<SafeAreaView style={{ flex: 1, backgroundColor: c.bg }}><StatusBar style="dark" />
    {screen.name === 'login' && <Login onDone={home} />}
    {screen.name === 'home' && <Home go={setScreen} signOut={() => api.logout().finally(() => setScreen({ name: 'login' }))} />}
    {screen.name === 'scan' && <Scan done={sc => setScreen({ name: 'result', scan: sc })} back={home} />}
    {screen.name === 'result' && <Result scan={screen.scan} back={home} />}
    {screen.name === 'help' && <Help back={() => setScreen(screen.from)} />}
    {screen.name !== 'help' && <Pressable accessibilityRole="button" accessibilityLabel="Help" onPress={() => setScreen({ name: 'help', from: screen })} style={s.fab}><Text style={s.fabText}>?</Text></Pressable>}
  </SafeAreaView>);
}

const s = StyleSheet.create({
  pad: { padding: 20, gap: 14 }, row: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 10 },
  h1: { fontFamily: f.display, fontSize: 28, color: c.navy, flexShrink: 1 }, h3: { fontFamily: f.bodyBold, fontSize: 16, color: c.navy },
  p: { fontFamily: f.body, fontSize: 15, lineHeight: 22, color: c.muted }, li: { fontFamily: f.body, fontSize: 14, lineHeight: 21, color: c.navy },
  fine: { fontFamily: f.body, fontSize: 12, lineHeight: 18, color: c.muted }, err: { fontFamily: f.body, color: c.sindoor, fontSize: 14 },
  big: { fontSize: 44, color: c.navy },
  input: { minHeight: 48, borderWidth: 1, borderColor: c.line, borderRadius: 10, backgroundColor: c.card, paddingHorizontal: 14, fontFamily: f.body, fontSize: 16, color: c.navy },
  btn: { minHeight: 48, borderRadius: 10, backgroundColor: c.teal, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 18 },
  btnGhost: { backgroundColor: 'transparent', borderWidth: 1, borderColor: '#9CC9CB' }, btnText: { fontFamily: f.bodyBold, color: '#fff', fontSize: 16 },
  card: { backgroundColor: c.card, borderRadius: 14, borderWidth: 1, borderColor: c.line, padding: 16, gap: 6 },
  chip: { minHeight: 48, borderRadius: 10, borderWidth: 1, borderColor: '#9CC9CB', paddingHorizontal: 16, alignItems: 'center', justifyContent: 'center', backgroundColor: '#fff' },
  chipOn: { backgroundColor: c.teal, borderColor: c.teal }, chipText: { fontFamily: f.bodyBold, fontSize: 16, color: c.teal },
  fab: { position: 'absolute', right: 16, bottom: 24, width: 56, height: 56, borderRadius: 28, backgroundColor: c.teal, alignItems: 'center', justifyContent: 'center', elevation: 6, shadowColor: '#0C1E36', shadowOpacity: 0.3, shadowRadius: 8, shadowOffset: { width: 0, height: 3 } },
  fabText: { color: '#fff', fontFamily: f.display, fontSize: 28 },
  badge: { borderRadius: 8, paddingHorizontal: 10, paddingVertical: 6 }, photo: { width: '100%', height: 260, borderRadius: 12, resizeMode: 'contain', backgroundColor: '#EDF3F0' },
});
