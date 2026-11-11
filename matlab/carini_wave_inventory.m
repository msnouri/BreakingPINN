%% carini_wave_inventory.m   (version 3, 10 Nov 2026)
% Inventory of the Duck 2016 lidar + infrared dataset (Carini, 2020; Carini
% et al., 2021a,b) for the field test of the breaking-inception parameter B.
%
% Changes from version 1, after the first run on the real files
%   - Wave types come from the authors' index lists (indNB, indSS, indSpill,
%     indPlunge). manTypecat holds one code per scan, so version 1, which read
%     only the first code of each wave, found 2 breakers instead of 270. If the
%     lists and the codes disagree, the types are taken from the codes.
%   - The observed onset is the first scan that manTypecat codes as spilling
%     (2) or plunging (3). The IR mask near the crest is kept as a check.
%   - The crest is placed on the profile grid by the first method that
%     reproduces the tracked crest (ixcat, xcat, x = 0, or the profile peak
%     equal to zcat); section 5 prints the test. In version 1, xcat never fell
%     on the profile grid.
%   - Depth is read from the row of hMSLcat for each wave, at the crest.
%   - Section 4 prints the raw contents of a few waves of each type.
%   - Warnings about function handles saved in the second file are silenced.
%
% Changes from version 2, after the second run
%   - The profiles are stored relative to the crest and the trough: x = 0 at
%     the crest (row ixcat) and z at the crest equals the wave height Hcat. The
%     crest test now accepts a match to zcat or to Hcat, or a local maximum.
%   - Breakers whose code sequence is a scan shorter or longer than the track
%     keep their manual onset (flagged in column codeLenDiff).
%   - Crest speed C is fitted to the tracked crest positions (ccat is NaN).
%
% What it does
%   1. Lists the variables in both .mat files (size, class, memory needed).
%   2. Reads the wave types and the per-scan manual codes, and finds the
%      onset scan of every breaker.
%   3. Loads the lidar profiles and the IR breaking mask.
%   4. Prints the raw contents of a few waves of each type.
%   5. Finds how the tracked crest maps onto the profile grid.
%   6. For every wave: scans and time recorded before onset, valid lidar
%      points near the crest, IR onset near the crest, depth, H, T and C.
%   7. Flags the breakers usable for the B test and the non-breaking waves
%      usable as controls; saves a table (CSV + MAT) and a log.
%   8. Figures: histograms and the lidar profiles of one breaker.
%   9. Lists the contents of the main structures in the second file.
%
% Data (CC BY), record http://hdl.handle.net/1773/46586
%   lidarIR_sharedData.mat (3.27 GB): per-wave profiles and labels
%   https://digital.lib.washington.edu/researchworks/bitstreams/7f9c23eb-654c-4706-adb8-1d7be96b504f/download
%   SurfZoneWavesAtOnsetBreaking.mat (2.65 GB): bathymetry, conditions, figure data
%   https://digital.lib.washington.edu/researchworks/bitstreams/f6481e57-5239-49f1-ae17-54f906c65503/download
%
% Notes
%   - Variables are loaded into structs, never into the workspace: the file
%     contains a variable named 'path', which would shadow MATLAB's path.
%   - Put both .mat files next to this script, or edit the paths below.
%     Run it, then send back carini_inventory_log.txt and the figures.

clear; clc; close all;

%% ---- Settings ------------------------------------------------------------
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end                 % when run cell by cell

fileWaves = fullfile(here, 'lidarIR_sharedData.mat');
fileFigs  = fullfile(here, 'SurfZoneWavesAtOnsetBreaking.mat');

opts.crestWindow    = 2.0;   % [m] half-width of the window around the crest
opts.minLeadT       = 0.25;  % [T] record needed before onset (inception precedes onset by a fraction of T)
opts.minValid       = 0.5;   % [-] median fraction of valid lidar points near the crest
opts.minTrackT      = 0.5;   % [T] track length needed for a non-breaking control wave
opts.zTol           = 0.05;  % [m] spread allowed between the profile z at the crest and zcat (section 5)
opts.nProbe         = 1;     % waves of each type printed in full in section 4
opts.exampleType    = 3;     % wave plotted in Figure 2: 3 = plunging, 2 = spilling
opts.makeFigures    = true;
opts.readSecondFile = true;  % list the structures of the second file (section 9)

% Optional: download from MATLAB instead of the browser (large files)
% wo = weboptions('Timeout', Inf);
% websave(fileWaves, 'https://digital.lib.washington.edu/researchworks/bitstreams/7f9c23eb-654c-4706-adb8-1d7be96b504f/download', wo);
% websave(fileFigs,  'https://digital.lib.washington.edu/researchworks/bitstreams/f6481e57-5239-49f1-ae17-54f906c65503/download', wo);

assert(isfile(fileWaves), 'Not found: %s', fileWaves);
logFile = fullfile(here, 'carini_inventory_log.txt');
if isfile(logFile), delete(logFile); end
diary(logFile);
fprintf('carini_wave_inventory.m, version 3, run %s\n', datestr(now, 'yyyy-mm-dd HH:MM'));

%% ---- 1. What is in the files ---------------------------------------------
W = listMatFile(fileWaves);
inFile = {W.name};
if isfile(fileFigs), listMatFile(fileFigs); end

%% ---- 2. Wave types, manual codes and onset --------------------------------
small = {'tcat', 'xcat', 'zcat', 'ixcat', 'itcat', 'Hcat', 'Tcat', 'hMSLcat', ...
         'manTypecat', 'ztrprecat', ...
         'indNB', 'indSS', 'indSpill', 'indPlunge', ...
         'countNB', 'countSS', 'countSpill', 'countPlunge', 'countTotal'};
small = small(ismember(small, inFile));
S = load(fileWaves, small{:});
assert(all(isfield(S, {'tcat', 'zcat', 'manTypecat'})), ...
    'tcat, zcat or manTypecat is missing: check the variable list above.');

nW  = numel(S.tcat);
nTk = zeros(nW, 1);
for k = 1:nW, nTk(k) = numel(cellOf(S.tcat, k)); end
fprintf('\n%d tracked waves, %d to %d scans each (median %d)\n', nW, min(nTk), max(nTk), round(median(nTk)));
if isfield(S, 'countTotal'), fprintf('  countTotal in the file: %g\n', double(S.countTotal)); end

% wave type from the authors' index lists (-1 = in no list)
cls = -ones(nW, 1);
indNames = {'indNB', 'indSS', 'indSpill', 'indPlunge'};
cntNames = {'countNB', 'countSS', 'countSpill', 'countPlunge'};
nTwice = 0;
for c = 0:3
    if ~isfield(S, indNames{c+1}), continue; end
    ii = double(S.(indNames{c+1})(:));
    ok = ~isempty(ii) && all(ii >= 1 & ii <= nW & ii == round(ii));
    cnt = NaN;
    if isfield(S, cntNames{c+1}), cnt = double(S.(cntNames{c+1})); end
    fprintf('  %-10s %5d entries (%s = %g), all valid wave numbers: %d\n', ...
        indNames{c+1}, numel(ii), cntNames{c+1}, cnt, ok);
    if ~ok, continue; end
    nTwice = nTwice + sum(cls(ii) >= 0);
    cls(ii) = c;
end
fprintf('  waves in no list: %d; waves in more than one list: %d\n', sum(cls < 0), nTwice);

% manual codes, one per scan
codes = cell(nW, 1);
for k = 1:nW
    v = cellOf(S.manTypecat, k);
    codes{k} = double(v(:));
end
perScan = cellfun(@numel, codes) == nTk;
codePat = repmat({'other'}, nW, 1);
for k = 1:nW
    v = codes{k};
    if isempty(v),                    codePat{k} = 'empty';
    elseif all(v == 0),               codePat{k} = 'all 0';
    elseif all(v == 1),               codePat{k} = 'all 1';
    elseif any(v == 3),               codePat{k} = 'has 3';
    elseif any(v == 2),               codePat{k} = 'has 2';
    elseif all(ismember(v, [0 1])),   codePat{k} = '0 and 1';
    end
end
fprintf('\nmanTypecat: one code per scan in %d of %d waves\n', sum(perScan), nW);
pats = {'all 0', 'all 1', '0 and 1', 'has 2', 'has 3', 'other', 'empty'};
fprintf('  %-14s', 'type \ codes');
fprintf('%9s', pats{:});
fprintf('\n');
for c = -1:3
    if ~any(cls == c), continue; end
    fprintf('  %-14s', typeName(c));
    for p = 1:numel(pats), fprintf('%9d', sum(cls == c & strcmp(codePat, pats{p}))); end
    fprintf('\n');
end
fprintf('  code sequences (code x number of scans), first waves of each type:\n');
for c = -1:3
    for k = reshape(find(cls == c, 3), 1, [])
        fprintf('    wave %4d %-13s %s\n', k, typeName(c), rle(codes{k}));
    end
end

% the lists and the codes should agree; if they do not, classify from the codes
nListed = sum(cls >= 2);
agree = sum((cls == 2 & strcmp(codePat, 'has 2')) | (cls == 3 & strcmp(codePat, 'has 3')));
fprintf('  listed breakers whose codes contain their type: %d of %d\n', agree, nListed);
if all(cls < 0) || (nListed > 0 && agree < 0.8 * nListed)
    fprintf('  -> the index lists do not match manTypecat: wave types taken from the codes instead\n');
    cls = -ones(nW, 1);
    cls(strcmp(codePat, 'all 0')) = 0;
    cls(strcmp(codePat, 'all 1')) = 1;
    cls(strcmp(codePat, 'has 2')) = 2;
    cls(strcmp(codePat, 'has 3')) = 3;
    fprintf('     nonbreaking %d, steady state %d, spilling %d, plunging %d, unclassified %d\n', ...
        sum(cls == 0), sum(cls == 1), sum(cls == 2), sum(cls == 3), sum(cls < 0));
end

% onset = first scan coded as the wave's breaker type (2 or 3)
lenDiff = cellfun(@numel, codes) - nTk;            % codes minus scans (0 for most waves)
onMan = nan(nW, 1);
for k = 1:nW
    if cls(k) >= 2 && abs(lenDiff(k)) <= 2
        i = find(codes{k} == cls(k), 1);
        if isempty(i), i = find(codes{k} >= 2, 1); end
        if ~isempty(i) && i <= nTk(k), onMan(k) = i; end
    end
end
isBrk = cls >= 2;
fprintf('  breakers (spilling + plunging): %d; onset scan found in manTypecat: %d; onset at the first scan: %d\n', ...
    sum(isBrk), sum(isBrk & ~isnan(onMan)), sum(onMan == 1));
fprintf('  breakers whose code sequence is 1-2 entries off the scan count (onset kept, flagged): %d\n', ...
    sum(isBrk & lenDiff ~= 0 & ~isnan(onMan)));

%% ---- 3. Access to the profiles (largest variables) -----------------------
big = {'xprofilesCat', 'zprofilesCat', 'profilesMaskCat'};
assert(all(ismember(big, inFile)), 'Profile variables missing: check the variable list above.');
src.isV73 = isV73(fileWaves);
if src.isV73
    src.m = matfile(fileWaves);
    for i = 1:numel(big)
        src.sz.(big{i}) = size(src.m, big{i});
        if prod(src.sz.(big{i})) ~= nW
            warning('%s has %d cells for %d waves.', big{i}, prod(src.sz.(big{i})), nW);
        end
    end
    fprintf('\nMAT v7.3: profiles are read one wave at a time.\n');
else
    fprintf('\nMAT v7 or older: loading the profiles whole (%.1f GB in memory) ...\n', ...
        sum([W(ismember(inFile, big)).bytes]) / 2^30);
    tic;
    src.data = load(fileWaves, big{:});
    fprintf('  loaded in %.0f s\n', toc);
end

%% ---- 4. Raw contents of a few waves -----------------------------------------
t0 = nan(nW, 1);
for k = 1:nW
    tk = toSeconds(cellOf(S.tcat, k));
    if ~isempty(tk), t0(k) = tk(1); end
end
dayNo = floor(t0 / 86400);                         % UTC day of each wave
probe = [];
for c = [2 3 0 1 -1]
    probe = [probe; find(cls == c, opts.nProbe)]; %#ok<AGROW>
end
for d = reshape(unique(dayNo(isfinite(dayNo))), 1, [])
    probe = [probe; find(dayNo == d & isBrk, 1)]; %#ok<AGROW>     first breaker of each day
end
probe = unique(probe, 'stable');
fprintf('\nRaw contents of %d waves:\n', numel(probe));
for k = reshape(probe, 1, [])
    probeWave(k, cls(k), codes{k}, S, src, nW);
end

%% ---- 5. Where is the crest on the profile grid? --------------------------------
% A method passes if, on the rows it picks, the profile elevation matches the tracked crest
% elevation zcat or the wave height Hcat (profiles measured from the trough), with a spread
% <= opts.zTol, or if the row is a crest in >= 80% of the scans: the highest point within
% +-0.5 m, standing at least 2 cm above the lowest point within +-2 m. The last method matches zcat by construction and passes only if the x shift between
% the grid and xcat stays constant along each track. The first method that passes is used.
crestMethods = {'ixcat', 'xcat', 'xcat in cm', 'x = 0', 'zcat match'};
crestDesc    = {'row index ixcat', 'nearest x to xcat', 'nearest x/100 to xcat', ...
                'nearest x to 0 (crest-relative grid)', 'local maximum of z equal to zcat'};
sampleW = find(cls >= 0);
sampleW = sampleW(unique(round(linspace(1, numel(sampleW), min(60, numel(sampleW))))));
fprintf('\nPlacing the tracked crest on the profile grid (%d waves tested; spreads in m):\n', numel(sampleW));
fprintf('  %-38s %8s %10s %10s %11s %18s\n', 'method', 'located', 'z - zcat', 'z - H', 'local max', 'x - xcat (spread)');
nM = numel(crestMethods);
pass = false(1, nM);
spreadZ = nan(1, nM);
spreadH = nan(1, nM);
for m = 1:nM
    dzc = [];
    dzh = [];
    offMed = [];
    offSpread = [];
    nLoc = 0;
    nAll = 0;
    nMax = 0;
    for k = reshape(sampleW, 1, [])
        nT = nTk(k);
        X  = readWave(src, 'xprofilesCat', k);
        Z  = readWave(src, 'zprofilesCat', k);
        r  = crestRows(crestMethods{m}, S, k, X, Z, nT);
        zc = fieldVec(S, 'zcat', k);
        xc = fieldVec(S, 'xcat', k);
        Hk = fieldVec(S, 'Hcat', k);
        nAll = nAll + nT;
        offk = [];
        for j = 1:nT
            if isnan(r(j)), continue; end
            nLoc = nLoc + 1;
            z = getScan(Z, j, nT);
            x = getScan(X, j, nT);
            lo = max(1, r(j) - 50);                 % +-0.5 m: highest point
            hi = min(numel(z), r(j) + 50);
            lo2 = max(1, r(j) - 200);               % +-2 m: at least 2 cm above the lowest point
            hi2 = min(numel(z), r(j) + 200);
            if isfinite(z(r(j))) && z(r(j)) >= max(z(lo:hi)) - 1e-9 && z(r(j)) - min(z(lo2:hi2)) >= 0.02
                nMax = nMax + 1;
            end
            dzc(end+1) = z(r(j)) - at(zc, j); %#ok<AGROW>
            dzh(end+1) = z(r(j)) - at(Hk, j); %#ok<AGROW>
            offk(end+1) = x(r(j)) - at(xc, j); %#ok<AGROW>
        end
        offk = offk(isfinite(offk));
        if numel(offk) >= 3
            offMed(end+1)    = median(offk); %#ok<AGROW>
            offSpread(end+1) = median(abs(offk - median(offk))); %#ok<AGROW>
        end
    end
    fLoc = nLoc / max(nAll, 1);
    fMax = nMax / max(nLoc, 1);
    spreadZ(m) = madOf(dzc);
    spreadH(m) = madOf(dzh);
    if m < nM
        pass(m) = fLoc >= 0.5 && (spreadZ(m) <= opts.zTol || spreadH(m) <= opts.zTol || fMax >= 0.8);
    else
        pass(m) = fLoc >= 0.5 && median(offSpread) <= 0.10;
    end
    fprintf('  %-38s %7.0f%% %10.4f %10.4f %10.0f%% %10.2f (%5.2f)\n', crestDesc{m}, 100 * fLoc, ...
        spreadZ(m), spreadH(m), 100 * fMax, median(offMed), median(offSpread));
end
mBest = find(pass, 1);
if isempty(mBest)
    fprintf(['\nNo method places the tracked crest on the profile grid.\n', ...
        'Stopping here: please send carini_inventory_log.txt (section 4 shows how the profiles are stored).\n']);
    diary off;
    error('inventory:crest', 'Crest not found on the profile grid: see the log.');
end
crestMethod = crestMethods{mBest};
fprintf('  -> using %s\n', crestDesc{mBest});
if spreadH(mBest) <= opts.zTol
    zRef = 'trough (z at the crest = Hcat)';
elseif spreadZ(mBest) <= opts.zTol
    zRef = 'same datum as zcat';
else
    zRef = 'unknown (crest accepted as the local maximum)';
end
fprintf('  vertical reference of the profiles: %s\n', zRef);

%% ---- 6. Per-wave inventory ------------------------------------------------------
R = repmat(struct('tStart', NaN, 'nScans', NaN, 'rateHz', NaN, 'trackDur', NaN, 'T', NaN, ...
    'H', NaN, 'h', NaN, 'C', NaN, 'onsetMan', NaN, 'onsetIR', NaN, 'onsetIRany', NaN, ...
    'onset', NaN, 'onsetSrc', 0, 'codeLenDiff', 0, 'nPre', NaN, 'tPre', NaN, 'xStart', NaN, 'xOnset', NaN, ...
    'validCrest', NaN, 'validAll', NaN, 'nGoodPre', NaN), nW, 1);
tic;
fprintf('\nInventory of %d waves ...\n', nW);
for k = 1:nW
    t  = toSeconds(cellOf(S.tcat, k));
    nT = numel(t);
    if nT == 0, continue; end
    X = readWave(src, 'xprofilesCat', k);
    Z = readWave(src, 'zprofilesCat', k);
    M = readWave(src, 'profilesMaskCat', k);
    r = crestRows(crestMethod, S, k, X, Z, nT);
    w = windowPoints(getScan(X, 1, nT), opts.crestWindow, crestMethod);

    [fCrest, fAll] = validFractions(Z, r, w, nT);
    [iIR, iAny]    = irOnset(M, r, w, nT);

    iOn = onMan(k);                                  % authors' manual onset
    src1 = 1;
    if isBrk(k) && isnan(iOn) && ~isnan(iIR)         % no manual onset: fall back on the IR mask
        iOn  = iIR;
        src1 = 2;
    end

    R(k).tStart     = t(1);
    R(k).nScans     = nT;
    R(k).rateHz     = 1 / median(diff(t));
    R(k).trackDur   = t(end) - t(1);
    R(k).onsetMan   = onMan(k);
    R(k).onsetIR    = iIR;
    R(k).onsetIRany = iAny;
    R(k).codeLenDiff = lenDiff(k);
    xc = fieldVec(S, 'xcat', k);
    if ~isempty(xc), R(k).xStart = xc(1); end
    R(k).T = valueAt(fieldVec(S, 'Tcat', k), NaN, nT);
    R(k).H = valueAt(fieldVec(S, 'Hcat', k), iOn, nT);
    R(k).C = crestSpeed(t, xc, iOn);
    R(k).h = depthAtCrest(depthRow(S, k, nW), r, iOn);

    if isBrk(k) && ~isnan(iOn)
        R(k).onset    = iOn;
        R(k).onsetSrc = src1;
        pre = 1:iOn-1;                               % scans recorded before onset
        R(k).nPre = numel(pre);
        R(k).tPre = t(iOn) - t(1);
        if iOn <= numel(xc), R(k).xOnset = xc(iOn); end
    else
        pre = 1:nT;                                  % whole track
    end
    if ~isempty(pre)
        R(k).validCrest = median(fCrest(pre), 'omitnan');
        R(k).validAll   = median(fAll(pre), 'omitnan');
        R(k).nGoodPre   = sum(fCrest(pre) >= opts.minValid);
    else
        R(k).nGoodPre = 0;
    end

    if mod(k, 250) == 0, fprintf('  %d of %d waves (%.0f s)\n', k, nW, toc); end
end

%% ---- 7. Table, summary and files -------------------------------------------
names = arrayfun(@typeName, cls, 'UniformOutput', false);
Tb = [table((1:nW)', cls, categorical(names), 'VariableNames', {'wave', 'type', 'typeName'}), ...
      struct2table(R)];
Tb.tPre_T  = Tb.tPre ./ Tb.T;
Tb.track_T = Tb.trackDur ./ Tb.T;
Tb.H_h     = Tb.H ./ Tb.h;
lead = Tb.tPre_T >= opts.minLeadT;
good = Tb.validCrest >= opts.minValid;
Tb.usable = (isBrk & lead & good) | (cls == 0 & Tb.track_T >= opts.minTrackT & good);

fprintf('\n================ SUMMARY ================\n');
fprintf('Crest placed on the profile grid with: %s; profiles measured from: %s\n', crestDesc{mBest}, zRef);
fprintf('Median scan rate %.2f Hz (expected 5.2 Hz); median wave period %.1f s; median track %.2f T\n', ...
    median(Tb.rateHz, 'omitnan'), median(Tb.T, 'omitnan'), median(Tb.track_T, 'omitnan'));
fprintf('Waves: nonbreaking %d, steady state %d, spilling %d, plunging %d, in no list %d\n', ...
    sum(cls == 0), sum(cls == 1), sum(cls == 2), sum(cls == 3), sum(cls < 0));
fprintf('Breakers with observed onset: %d\n', sum(isBrk));
fprintf('  onset from manTypecat %d, from the IR mask (no manual onset) %d, none %d\n', ...
    sum(isBrk & Tb.onsetSrc == 1), sum(isBrk & Tb.onsetSrc == 2), sum(isBrk & isnan(Tb.onset)));
fprintf('  onset at the first scan (nothing recorded before onset): %d\n', sum(isBrk & Tb.onset == 1));
fprintf('  code sequence 1-2 entries off the scan count (onset may be off by a scan): %d\n', ...
    sum(isBrk & Tb.codeLenDiff ~= 0 & Tb.onsetSrc == 1));
both = isBrk & ~isnan(Tb.onsetMan) & ~isnan(Tb.onsetIR);
dOn  = Tb.onsetIR(both) - Tb.onsetMan(both);
fprintf(['  IR mask near the crest vs manual onset (%d breakers): same scan %d, within one scan %d, ', ...
    'IR earlier %d, IR later %d; no IR flag near the crest %d\n'], sum(both), sum(dOn == 0), ...
    sum(abs(dOn) <= 1), sum(dOn < 0), sum(dOn > 0), sum(isBrk & ~isnan(Tb.onsetMan) & isnan(Tb.onsetIR)));
for c = [2 3]
    sel = cls == c & ~isnan(Tb.onset);
    q  = pct(Tb.tPre_T(sel), [10 50 90]);
    hq = pct(Tb.h(sel), [10 50 90]);
    fprintf('  %-9s record before onset: median %.2f T (10th-90th pct %.2f-%.2f T), median %g scans\n', ...
        typeName(c), q(2), q(1), q(3), round(median(Tb.nPre(sel), 'omitnan')));
    fprintf('            depth at onset: median %.2f m (10th-90th pct %.2f-%.2f m); H/h median %.2f; C median %.2f m/s\n', ...
        hq(2), hq(1), hq(3), median(Tb.H_h(sel), 'omitnan'), median(Tb.C(sel), 'omitnan'));
    fprintf('            usable for the B test (>= %.2f T before onset, >= %.0f%% valid near crest): %d of %d\n', ...
        opts.minLeadT, 100*opts.minValid, sum(Tb.usable & cls == c), sum(cls == c));
end
fprintf('  breakers failing only the record test: %d; only the lidar-coverage test: %d; both: %d\n', ...
    sum(isBrk & ~lead & good), sum(isBrk & lead & ~good), sum(isBrk & ~lead & ~good));
fprintf('  usable breakers vs the record required before onset:');
for L = [0.10 0.15 0.20 0.25 0.30 0.40 0.50]
    fprintf('  %.2fT: %d', L, sum(isBrk & Tb.tPre_T >= L & good));
end
fprintf('\n');
fprintf('Non-breaking waves usable as controls (>= %.2f T tracked): %d of %d\n', ...
    opts.minTrackT, sum(Tb.usable & cls == 0), sum(cls == 0));
fprintf('Non-breaking waves with IR breaking flagged near the crest (check these): %d\n', ...
    sum(cls == 0 & ~isnan(Tb.onsetIR)));
fprintf('Median valid fraction near the crest before onset: spilling %.2f, plunging %.2f, non-breaking %.2f\n', ...
    median(Tb.validCrest(cls == 2), 'omitnan'), median(Tb.validCrest(cls == 3), 'omitnan'), ...
    median(Tb.validCrest(cls == 0), 'omitnan'));

writetable(Tb, fullfile(here, 'carini_wave_inventory.csv'));
save(fullfile(here, 'carini_wave_inventory.mat'), 'Tb', 'opts', 'crestMethod');
fprintf('\nSaved carini_wave_inventory.csv / .mat in %s\n', here);

%% ---- 8. Figures --------------------------------------------------------------
if opts.makeFigures
    fig1 = newFigure([100 100 1600 520]);
    tl = tiledlayout(fig1, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    ax = nexttile(tl);
    histByType(ax, clipHigh(Tb.tPre_T, 1.999), cls, [2 3], 0:0.1:2);
    xline(ax, opts.minLeadT, '--k', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    xlabel(ax, 'Record before onset, $t_{\mathrm{pre}}/T$');
    ylabel(ax, 'Number of breakers');
    xticks(ax, 0:0.5:2);
    panelLabel(ax, '(a)');

    ax = nexttile(tl);
    histByType(ax, clipHigh(Tb.nPre, 49), cls, [2 3], 0:2:50);
    xlabel(ax, 'Lidar scans before onset');
    ylabel(ax, 'Number of breakers');
    xticks(ax, 0:10:50);
    panelLabel(ax, '(b)');

    ax = nexttile(tl);
    histByType(ax, Tb.validCrest, cls, [0 2 3], 0:0.05:1);
    xline(ax, opts.minValid, '--k', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    xlabel(ax, 'Valid fraction near the crest');
    ylabel(ax, 'Number of waves');
    xticks(ax, 0:0.2:1);
    panelLabel(ax, '(c)');

    exportFig(fig1, fullfile(here, 'carini_inventory_hist'));

    cand = find(Tb.usable & cls == opts.exampleType);
    if ~isempty(cand)
        [~, j] = max(Tb.tPre_T(cand));
        plotExampleWave(src, S, cand(j), Tb, here, crestMethod);
    else
        fprintf('No usable wave of type %d to plot.\n', opts.exampleType);
    end
end

%% ---- 9. Contents of the second file --------------------------------------------
if opts.readSecondFile && isfile(fileFigs)
    ws = warning('off', 'all');                    % function handles saved by the authors
    G = load(fileFigs, 'WaveCond');
    if isfield(G, 'WaveCond')
        wc = G.WaveCond(1);
        if all(isfield(wc, {'FRFadop_Hs', 'FRFadop_Tp'}))
            fprintf(['\nWaveCond: Hs %.2f-%.2f m, Tp %.1f-%.1f s over the stored record ', ...
                '(the lidar runs had 2.0-2.5 m and 7.0-14.5 s)\n'], ...
                min(wc.FRFadop_Hs), max(wc.FRFadop_Hs), min(wc.FRFadop_Tp), max(wc.FRFadop_Tp));
        end
    end
    % the bathymetry (FieldSetupBathy, 2.2 GB) was listed in the first run; BrkDetectEx (11 GB) is skipped
    for v = {'TemporalSpatialMethod', 'SpeedDist', 'SlopeMetric', 'DA_SP', 'DA_NB', ...
             'Prevalence_NBSP', 'FoamCov', 'Tracking'}
        try
            Gv = load(fileFigs, v{1});
            if isfield(Gv, v{1}), describe(v{1}, Gv.(v{1})); end
        catch err
            fprintf('\n%s could not be read: %s\n', v{1}, err.message);
        end
    end
    warning(ws);
end

diary off;
fprintf('\nLog written to %s\n', logFile);


%% ======================== local functions =====================================

function nm = typeName(c)
% Name of a wave type code (-1 = in no index list).
    allNames = {'unclassified', 'nonbreaking', 'steady state', 'spilling', 'plunging'};
    nm = allNames{c + 2};
end

function W = listMatFile(f)
% Prints the variables of a MAT file with the memory each needs once loaded.
    if isV73(f), fmt = 'MAT v7.3, partial loading possible'; else, fmt = 'MAT v7 or older'; end
    fprintf('\n%s  (%s)\n', f, fmt);
    ws = warning('off', 'all');                    % function handles saved by the authors
    W = whos('-file', f);
    warning(ws);
    fprintf('  %-22s %-18s %-10s %12s\n', 'variable', 'size', 'class', 'RAM [MB]');
    for i = 1:numel(W)
        fprintf('  %-22s %-18s %-10s %12.1f\n', W(i).name, mat2str(W(i).size), W(i).class, W(i).bytes / 2^20);
    end
    fprintf('  total once loaded: %.2f GB\n', sum([W.bytes]) / 2^30);
end

function tf = isV73(f)
% True for HDF5-based (v7.3) MAT files, which allow partial loading.
    fid = fopen(f, 'r');
    if fid < 0, error('isV73:open', 'Cannot open %s', f); end
    hdr = fread(fid, [1 116], '*char');
    fclose(fid);
    tf = contains(hdr, 'MATLAB 7.3');
end

function A = readWave(src, name, k)
% Contents of cell k of the per-wave variable NAME (one cell per tracked wave).
    if src.isV73
        sz = src.sz.(name);
        if sz(1) == 1, r = 1; c = k; else, r = k; c = 1; end
        switch name
            case 'xprofilesCat',    C = src.m.xprofilesCat(r, c);
            case 'zprofilesCat',    C = src.m.zprofilesCat(r, c);
            case 'profilesMaskCat', C = src.m.profilesMaskCat(r, c);
            otherwise, error('readWave:name', 'Unknown variable %s', name);
        end
    else
        C = src.data.(name)(k);
    end
    if iscell(C), A = C{1}; else, A = C; end
end

function A = cellOf(V, k)
% Element k of a per-wave variable, whether it is a cell array or numeric.
    if iscell(V), A = V{k}; else, A = V(k); end
end

function v = fieldVec(S, name, k)
% Values of field NAME for wave k as a column ([] if the field is missing).
    if isfield(S, name)
        v = double(cellOf(S.(name), k));
        v = v(:);
    else
        v = [];
    end
end

function v = at(a, j)
% a(j), or NaN when a has fewer than j values.
    if j <= numel(a), v = a(j); else, v = NaN; end
end

function d = madOf(v)
% Median absolute deviation of the finite values of v (NaN if there are none).
    v = v(isfinite(v));
    if isempty(v), d = NaN; else, d = median(abs(v - median(v))); end
end

function C = crestSpeed(t, xc, i)
% Crest speed |dx/dt| from a straight-line fit to the tracked crest positions:
% 5 scans around scan i, or the whole track when i is NaN.
    C = NaN;
    n = min(numel(t), numel(xc));
    if n < 3, return; end
    if isnan(i), idx = 1:n; else, idx = max(1, i - 2):min(n, i + 2); end
    idx = idx(isfinite(xc(idx)) & isfinite(t(idx)));
    if numel(idx) < 3, return; end
    p = polyfit(t(idx) - t(idx(1)), xc(idx), 1);
    C = abs(p(1));
end

function s = ends(a)
% 'first .. last (n values)' for printing.
    if isempty(a), s = '[]'; return; end
    if numel(a) == 1, s = sprintf('%.3f', a); return; end
    s = sprintf('%.3f .. %.3f (%d values)', a(1), a(end), numel(a));
end

function s = rle(v)
% Run-length text of a code sequence: '0x12 2x1 1x18'.
    v = v(:).';
    if isempty(v), s = '[]'; return; end
    b = [1, find(diff(v) ~= 0) + 1];
    n = diff([b, numel(v) + 1]);
    parts = arrayfun(@(a, m) sprintf('%gx%d', a, m), v(b), n, 'UniformOutput', false);
    s = strjoin(parts, ' ');
end

function v = getScan(P, j, nT)
% Profile of scan j from a per-wave array stored as a cell of scans,
% an nT-by-nX or nX-by-nT matrix, or a single grid shared by all scans.
    if iscell(P)
        v = P{j};
    elseif isvector(P)
        v = P;
    elseif size(P, 1) == nT
        v = P(j, :);
    elseif size(P, 2) == nT
        v = P(:, j);
    else
        error('getScan:size', 'Cannot match an array of size %s to %d scans.', mat2str(size(P)), nT);
    end
    v = double(v(:)).';
end

function h = depthRow(S, k, nW)
% Depth along the profile grid for wave k (hMSLcat holds one row per wave).
    h = [];
    if ~isfield(S, 'hMSLcat'), return; end
    D = S.hMSLcat;
    if iscell(D)
        h = double(D{k});
    elseif size(D, 1) == nW
        h = double(D(k, :));
    elseif size(D, 2) == nW
        h = double(D(:, k));
    else
        h = double(D(k));
    end
    h = h(:);
end

function h = depthAtCrest(hr, r, iOn)
% Depth under the crest at onset, or the median along the track without onset.
    h = NaN;
    if isempty(hr), return; end
    ok = ~isnan(r) & r >= 1 & r <= numel(hr);
    if ~isnan(iOn)
        if iOn <= numel(r) && ok(iOn), h = hr(r(iOn)); end
    elseif any(ok)
        h = median(hr(r(ok)), 'omitnan');
    end
end

function r = crestRows(method, S, k, X, Z, nT)
% Row of the tracked crest on the profile grid for each scan (NaN where not found).
    r = nan(nT, 1);
    switch method
        case 'zcat match'                          % local maximum (+-50 points) with z = zcat
            zc = fieldVec(S, 'zcat', k);
            for j = 1:min(nT, numel(zc))
                z = getScan(Z, j, nT);
                bestD = inf;
                for i = find(abs(z - zc(j)) <= 0.005)
                    lo = max(1, i - 50);
                    hi = min(numel(z), i + 50);
                    if z(i) >= max(z(lo:hi)) && abs(z(i) - zc(j)) < bestD
                        r(j) = i;
                        bestD = abs(z(i) - zc(j));
                    end
                end
            end
        case 'ixcat'
            if ~isfield(S, 'ixcat'), return; end
            ix = fieldVec(S, 'ixcat', k);
            nX = numel(getScan(X, 1, nT));
            n  = min(nT, numel(ix));
            r(1:n) = ix(1:n);
            bad = ~(r >= 1 & r <= nX & r == round(r));
            r(bad) = NaN;
        case {'xcat', 'xcat in cm', 'x = 0'}
            if strcmp(method, 'x = 0')
                xc = zeros(nT, 1);
            elseif isfield(S, 'xcat')
                xc = fieldVec(S, 'xcat', k);
            else
                return
            end
            scale = 1;
            if strcmp(method, 'xcat in cm'), scale = 0.01; end
            for j = 1:min(nT, numel(xc))
                x = getScan(X, j, nT) * scale;
                [dmin, i] = min(abs(x - xc(j)));
                dx = median(abs(diff(x)), 'omitnan');
                if dmin <= max(2 * dx, 1e-6), r(j) = i; end
            end
    end
end

function w = windowPoints(x, halfWidth, method)
% Number of grid points in HALFWIDTH metres on the profile grid x.
    dx = median(abs(diff(x)), 'omitnan');
    if strcmp(method, 'xcat in cm') || dx >= 0.5   % grid stored in cm
        dx = dx / 100;
    end
    if ~isfinite(dx) || dx <= 0, dx = 0.01; end     % 1 cm grid (README)
    w = max(1, round(halfWidth / dx));
end

function [fCrest, fAll] = validFractions(Z, r, w, nT)
% Fraction of finite lidar points within w grid points of the crest, and on the whole profile.
    fCrest = nan(nT, 1);
    fAll   = nan(nT, 1);
    for j = 1:nT
        z  = getScan(Z, j, nT);
        ok = isfinite(z);
        fAll(j) = mean(ok);
        if j <= numel(r) && ~isnan(r(j))
            rows = max(1, r(j) - w):min(numel(z), r(j) + w);
            fCrest(j) = mean(ok(rows));
        end
    end
end

function [iCrest, iAny] = irOnset(M, r, w, nT)
% First scan with IR breaking within w grid points of the crest (iCrest) and anywhere (iAny).
    iCrest = NaN;
    iAny   = NaN;
    if ~iscell(M) && isvector(M) && numel(M) == nT && nT > 1
        i = find(double(M(:)) > 0, 1);             % one flag per scan
        if ~isempty(i), iCrest = i; iAny = i; end
        return
    end
    for j = 1:nT
        m = getScan(M, j, nT) > 0;
        if ~any(m), continue; end
        if isnan(iAny), iAny = j; end
        if j <= numel(r) && ~isnan(r(j))
            rows = max(1, r(j) - w):min(numel(m), r(j) + w);
            if any(m(rows)), iCrest = j; return; end
        end
    end
end

function v = valueAt(V, idx, nT)
% Value at scan IDX if V has one value per scan, otherwise the median of V.
    V = double(V(:));
    if numel(V) == nT && ~isnan(idx)
        v = V(idx);
    else
        v = median(V(isfinite(V)));
    end
    if isempty(v), v = NaN; end
end

function t = toSeconds(t)
% Epoch time in seconds (also accepts milliseconds or datenum days).
    t = double(t(:));
    tm = median(t, 'omitnan');
    if tm > 1e11
        t = t / 1000;                                % milliseconds
    elseif tm > 7e5 && tm < 1e6
        t = t * 86400;                               % datenum days
    end
end

function q = pct(x, p)
% Percentiles P of X (no toolbox needed).
    x = sort(x(isfinite(x)));
    if isempty(x), q = nan(size(p)); return; end
    if numel(x) == 1, q = repmat(x, size(p)); return; end
    q = interp1(linspace(0, 100, numel(x)), x, p);
end

function v = clipHigh(v, hi)
% Values above HI set to HI (to collect them in the last bin); NaN stays NaN.
    v(v > hi) = hi;
end

function probeWave(k, c, code, S, src, nW)
% Prints how one wave is stored: codes, crest track and the profile arrays.
    t  = toSeconds(cellOf(S.tcat, k));
    nT = numel(t);
    fprintf('\nWave %d (%s), %d scans, starts %s UTC\n', k, typeName(c), nT, ...
        datestr(t(1) / 86400 + datenum(1970, 1, 1), 'yyyy-mm-dd HH:MM:SS'));
    fprintf('  manTypecat  %s %s\n', mat2str(size(cellOf(S.manTypecat, k))), rle(code));
    if nT > 1, fprintf('  tcat        %.3f, step %.3f s\n', t(1), median(diff(t))); end
    xc = fieldVec(S, 'xcat', k);
    zc = fieldVec(S, 'zcat', k);
    for v = {'xcat', 'zcat', 'ixcat', 'itcat', 'Hcat', 'Tcat', 'ztrprecat'}
        if isfield(S, v{1})
            fprintf('  %-11s %-8s %s\n', v{1}, mat2str(size(cellOf(S.(v{1}), k))), ends(fieldVec(S, v{1}, k)));
        end
    end
    X = readWave(src, 'xprofilesCat', k);
    Z = readWave(src, 'zprofilesCat', k);
    M = readWave(src, 'profilesMaskCat', k);
    x1 = getScan(X, 1, nT);
    z1 = getScan(Z, 1, nT);
    fprintf('  x profile   %s %s; scan 1: %.3f to %.3f, step %.4f, NaN %.0f%%', class(X), mat2str(size(X)), ...
        x1(1), x1(end), median(diff(x1), 'omitnan'), 100 * mean(~isfinite(x1)));
    if ~isvector(X) && nT > 1
        dmax = 0;
        for j = 2:nT, dmax = max(dmax, max(abs(getScan(X, j, nT) - x1))); end
        fprintf('; largest change from scan 1: %.3f', dmax);
    end
    fprintf('\n');
    if ~isempty(xc)
        [dmin, i] = min(abs(x1 - xc(1)));
        fprintf('  x nearest to xcat(1) on scan 1: %.3f (%.3f away)\n', x1(i), dmin);
    end
    [zm, im] = max(z1);
    fprintf('  z profile   %s; scan 1: NaN %.0f%%, highest z %.3f at x = %.3f\n', mat2str(size(Z)), ...
        100 * mean(~isfinite(z1)), zm, x1(im));
    if isfield(S, 'ixcat')
        ix = fieldVec(S, 'ixcat', k);
        Hk = fieldVec(S, 'Hcat', k);
        tr = fieldVec(S, 'ztrprecat', k);
        for j = unique([1 nT])
            if j <= numel(ix) && ix(j) >= 1 && ix(j) <= numel(z1) && ix(j) == round(ix(j))
                xj = getScan(X, j, nT);
                zj = getScan(Z, j, nT);
                fprintf(['  scan %2d, row ixcat = %4d: x = %.3f, z = %.3f; zcat %.3f, Hcat %.3f, ', ...
                    'zcat - Hcat %.3f, ztrprecat %.3f\n'], j, ix(j), xj(ix(j)), zj(ix(j)), at(zc, j), ...
                    at(Hk, j), at(zc, j) - at(Hk, j), at(tr, j));
            end
        end
    end
    nOn = zeros(1, nT);
    for j = 1:nT, nOn(j) = sum(getScan(M, j, nT) > 0); end
    fprintf('  IR mask     %s %s; points flagged in each scan: %s\n', class(M), mat2str(size(M)), mat2str(nOn));
    j = find(nOn > 0, 1);
    if ~isempty(j)
        m  = getScan(M, j, nT) > 0;
        xj = getScan(X, j, nT);
        fprintf('              first flagged scan %d: x %.2f to %.2f\n', j, min(xj(m)), max(xj(m)));
    end
    hr = depthRow(S, k, nW);
    if ~isempty(hr), fprintf('  hMSLcat     row of %d: %s\n', numel(hr), ends(hr)); end
end

function describe(name, s)
% Field names, classes and sizes of a structure (two levels).
    fprintf('\n%s:\n', name);
    if ~isstruct(s) || isempty(s), fprintf('  %s %s\n', class(s), mat2str(size(s))); return; end
    if numel(s) > 1, fprintf('  struct array %s, first element:\n', mat2str(size(s))); end
    f = fieldnames(s);
    for i = 1:numel(f)
        v = s(1).(f{i});
        fprintf('  %-24s %-14s %s\n', f{i}, class(v), mat2str(size(v)));
        if isstruct(v) && ~isempty(v)
            g = fieldnames(v);
            for j = 1:numel(g)
                fprintf('      .%-20s %-14s %s\n', g{j}, class(v(1).(g{j})), mat2str(size(v(1).(g{j}))));
            end
        end
    end
end

function fig = newFigure(pos)
% White figure with LaTeX text and large fonts for every child object.
    fig = figure('Color', 'w', 'Position', pos);
    set(fig, 'DefaultTextInterpreter', 'latex', ...
             'DefaultAxesTickLabelInterpreter', 'latex', ...
             'DefaultLegendInterpreter', 'latex', ...
             'DefaultAxesFontSize', 18, ...
             'DefaultAxesLineWidth', 1.5, ...
             'DefaultLegendFontSize', 16);
end

function styleAxes(ax)
    box(ax, 'on');
    set(ax, 'LineWidth', 1.8, 'FontSize', 18, 'TickLabelInterpreter', 'latex', 'Layer', 'top');
end

function panelLabel(ax, s)
% Bold LaTeX panel label in the top-right corner.
    text(ax, 0.97, 0.95, ['\textbf{' s '}'], 'Units', 'normalized', 'Interpreter', 'latex', ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontSize', 20);
end

function histByType(ax, v, cls, types, edges)
% Overlaid histograms of V for the wave types in TYPES (0 = nonbreaking, 2 = spilling, 3 = plunging).
    cols = [0.50 0.50 0.50; 0.40 0.40 0.40; 0.00 0.45 0.74; 0.85 0.33 0.10];
    hold(ax, 'on');
    for i = 1:numel(types)
        x = v(cls == types(i) & isfinite(v));
        histogram(ax, x, edges, 'FaceColor', cols(types(i)+1, :), 'FaceAlpha', 0.55, ...
            'EdgeColor', 'none', 'DisplayName', typeName(types(i)));
    end
    legend(ax, 'Orientation', 'horizontal', 'Location', 'northoutside', 'Box', 'off');
    styleAxes(ax);
end

function plotExampleWave(src, S, k, Tb, here, crestMethod)
% Lidar profiles of wave k up to a few scans after onset, offset in time.
    t  = toSeconds(cellOf(S.tcat, k));
    nT = numel(t);
    X = readWave(src, 'xprofilesCat', k);
    Z = readWave(src, 'zprofilesCat', k);
    M = readWave(src, 'profilesMaskCat', k);
    r = crestRows(crestMethod, S, k, X, Z, nT);
    iOn  = Tb.onset(k);
    last = min(nT, iOn + 5);
    dz   = 0.15;                                     % offset between scans [m]

    fig = newFigure([150 150 950 850]);
    ax = axes(fig);
    hold(ax, 'on');
    xcr = nan(last, 1);
    for j = 1:last
        x = getScan(X, j, nT);
        z = getScan(Z, j, nT);
        off = (j - iOn) * dz;
        if j == iOn, col = [0.85 0.33 0.10]; lw = 2.5; else, col = [0.35 0.35 0.35]; lw = 1; end
        plot(ax, x, z + off, '-', 'Color', col, 'LineWidth', lw, 'HandleVisibility', 'off');
        m = getScan(M, j, nT) > 0;
        if numel(m) == numel(x) && any(m)
            plot(ax, x(m), z(m) + off, '.', 'Color', [0.80 0.10 0.10], 'MarkerSize', 8, 'HandleVisibility', 'off');
        end
        if ~isnan(r(j))
            xcr(j) = x(r(j));
            plot(ax, x(r(j)), z(r(j)) + off, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 4, 'HandleVisibility', 'off');
        end
    end
    plot(ax, NaN, NaN, '-', 'Color', [0.35 0.35 0.35], 'LineWidth', 1, 'DisplayName', 'scans');
    plot(ax, NaN, NaN, '-', 'Color', [0.85 0.33 0.10], 'LineWidth', 2.5, 'DisplayName', 'onset scan');
    plot(ax, NaN, NaN, '.', 'Color', [0.80 0.10 0.10], 'MarkerSize', 14, 'DisplayName', 'IR breaking');
    plot(ax, NaN, NaN, 'ko', 'MarkerFaceColor', 'k', 'DisplayName', 'tracked crest');
    legend(ax, 'Orientation', 'horizontal', 'Location', 'northoutside', 'NumColumns', 2, 'Box', 'off');
    xcr = xcr(isfinite(xcr));
    if numel(xcr) > 1 && xcr(end) < xcr(1)
        set(ax, 'XDir', 'reverse');                  % waves travel toward smaller x: offshore on the left
    end
    xlabel(ax, '$x$ on the profile grid (m)');
    ylabel(ax, '$z$ + offset of 0.15 m per scan (m)');
    title(ax, sprintf('Wave %d (%s): %d scans, %.2f$\\,T$ before onset', ...
        k, typeName(Tb.type(k)), Tb.nPre(k), Tb.tPre_T(k)), 'FontSize', 16);
    styleAxes(ax);
    exportFig(fig, fullfile(here, 'carini_example_wave'));
end

function exportFig(fig, base)
    try
        exportgraphics(fig, [base '.png'], 'Resolution', 200);
        exportgraphics(fig, [base '.pdf'], 'ContentType', 'vector');
    catch
        print(fig, [base '.png'], '-dpng', '-r200');
    end
end
