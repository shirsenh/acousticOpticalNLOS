clear;
info = audiodevinfo;
v = 340;
T = 1;
fs = 48000;
f1 = 20000;
f0 = 2000;
samples_per_chirp = uint32(fs*T/16);
t = linspace(0, T, uint32(fs*T));
y = chirp(t, f0, T, f1, 'linear');
y = reshape(y, [size(y, 2) 1]);

recObj = audiorecorder(fs, 16, 1, 6);
chirp = audioplayer(y, fs);
recordblocking(recObj, 3);
play(chirp);

rec = getaudiodata(recObj);
replay = audioplayer(rec, fs);
play(replay);