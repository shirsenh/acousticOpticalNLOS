clear;
info = audiodevinfo;
% recorder = audiorecorder(48000, 16, 1, 0);
v = 340;
T = 2;
fs = 48000;
f1 = 20000;
f0 = 2000;
samples_per_chirp = uint32(fs*T/16);
t = linspace(0, T, uint32(fs*T));
y = chirp(t, f0, T, f1, 'linear');
y = reshape(y, [size(y, 2) 1]);
playRec = audioPlayerRecorder(fs);
[rec, n, m] = playRec(y);
% audioObj = audioplayer(y, fs);
% audioObj.TimerFcn = 'showSeconds';
% audioObj.TimerPeriod = 2;
% recordblocking(recorder, 3);
% play(audioObj);
% disp('Press any key to play recorded sound');
% pause;
% play(recorder);
% y = getaudiodata(recorder);
% plot(y);