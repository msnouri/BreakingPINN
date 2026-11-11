%% carini_export_pinn.m   (version 2, 10 Nov 2026)
% Exports the Duck 2016 waves selected for the field test of the breaking-inception
% parameter B (Carini et al., 2021a,b) to one file that the Python PINN code can read.
%
% What it does
%   1. Reads the wave types and the onset scans as carini_wave_inventory.m (version 3).
%   2. Rebuilds, for every exported wave, the free surface on the fixed FRF grid. The
%      profiles are stored relative to the crest and to the preceding trough, so
%      x = xprofilesCat + xcat and eta = zprofilesCat + ztrprecat, in the datum of the
%      tracked crests.
%   3. Flags stretches of the gridded profiles that are straight lines over at least
%      opts.gapRunMin: linear interpolation across gaps in the lidar returns.
%   4. Collects still-water levels for every wave: tide gauge and lidar (WaveCond), the
%      level of the record (zMSLcat) and the mean of the gridded surface.
%   5. Checks steps 2 and 3 against the one record of the second file that keeps the raw
%      lidar returns (Tracking): how that record was gridded and how its gaps were filled,
%      the gaps near the crests tracked in it, and, for the waves of the per-wave file
%      recorded in it (matched by time), the rebuilt surface against the same returns.
%   6. Saves carini_pinn_input.mat (MAT v7, read with scipy.io.loadmat), the index
%      carini_pinn_index.csv and the log carini_export_log.txt.
%
% Exported waves
%   - breakers (spilling, plunging) with at least one scan before onset; retained = true
%     when the record before onset is at least opts.minLeadT periods;
%   - non-breaking waves tracked for at least opts.minTrackT periods (retained = true).
%
% In Python (carini_pinn_load.py, same folder):
%   from carini_pinn_load import load, training_points
%   waves, meta = load('carini_pinn_input.mat')
%
% Data (CC BY): http://hdl.handle.net/1773/46586. Put both .mat files next to this script.

clear; clc; close all;

%% ---- Settings ------------------------------------------------------------
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end

fileWaves = fullfile(here, 'lidarIR_sharedData.mat');
fileFigs  = fullfile(here, 'SurfZoneWavesAtOnsetBreaking.mat');
outFile   = fullfile(here, 'carini_pinn_input.mat');
csvFile   = fullfile(here, 'carini_pinn_index.csv');
logFile   = fullfile(here, 'carini_export_log.txt');

opts.minLeadT     = 0.25;  % [T] record before onset needed to retain a breaker
opts.minTrackT    = 0.5;   % [T] track length needed for a non-breaking control
opts.xStride      = 5;     % export every 5th point of the 1-cm grid (5 cm); 1 keeps them all
opts.gapRunMin    = 0.10;  % [m] straight stretches at least this long are flagged as gap fills
opts.straightTol  = 1e-9;  % [m] three points are collinear when |second difference| < this: exact linear
                           %     interpolation gives ~1e-13, a measured smooth surface ~1e-6 on a 1-cm grid
opts.crestWindow  = 2.0;   % [m] half-width of the crest region in the statistics
opts.dRaw         = 0.02;  % [m] a grid point is measured if a raw return lies within this distance
opts.checkExample = true;  % compare with the raw lidar of the example record (second file)

assert(exist(fileWaves, 'file') == 2, 'Not found: %s', fileWaves);
if exist(logFile, 'file') == 2, delete(logFile); end
diary(logFile);
fprintf('carini_export_pinn.m, version 2, run %s\n', datestr(now, 'yyyy-mm-dd HH:MM'));

%% ---- 1. Wave types, onset and record lengths -------------------------------
small = {'tcat', 'xcat', 'zcat', 'ixcat', 'itcat', 'Hcat', 'Tcat', 'ztrprecat', 'hMSLcat', ...
         'manTypecat', 'tstartcat', 'zMSLcat', 'indNB', 'indSS', 'indSpill', 'indPlunge'};
Wf = whos('-file', fileWaves);
small = small(ismember(small, {Wf.name}));
S = load(fileWaves, small{:});
need = {'tcat', 'xcat', 'zcat', 'ixcat', 'Hcat', 'Tcat', 'ztrprecat', 'hMSLcat', 'manTypecat', ...
        'indNB', 'indSpill', 'indPlunge'};
absent = need(~isfield(S, need));
assert(isempty(absent), 'Missing in %s: %s', fileWaves, strjoin(absent, ', '));

nW  = numel(S.tcat);
cls = -ones(nW, 1);                                % -1: in no class list
lists = {'indNB', 0; 'indSS', 1; 'indSpill', 2; 'indPlunge', 3};
for i = 1:size(lists, 1)
    if isfield(S, lists{i, 1}), cls(double(S.(lists{i, 1}))) = lists{i, 2}; end
end

t0 = nan(nW, 1);  nTk = zeros(nW, 1);  onset = nan(nW, 1);  lenDiff = zeros(nW, 1);
Tw = nan(nW, 1);  trackDur = nan(nW, 1);  tPre = nan(nW, 1);
for k = 1:nW
    t = toEpoch(cellOf(S.tcat, k));
    nTk(k) = numel(t);
    if nTk(k) == 0, continue; end
    t0(k) = t(1);
    trackDur(k) = t(end) - t(1);
    Tw(k) = medf(fieldVec(S, 'Tcat', k));
    code = fieldVec(S, 'manTypecat', k);
    lenDiff(k) = numel(code) - nTk(k);
    if cls(k) >= 2 && abs(lenDiff(k)) <= 2
        i = find(code == cls(k), 1);
        if isempty(i), i = find(code >= 2, 1); end
        if ~isempty(i) && i <= nTk(k)
            onset(k) = i;
            tPre(k)  = t(i) - t(1);
        end
    end
end
tPreT   = tPre ./ Tw;
trackT  = trackDur ./ Tw;
isBrk   = cls >= 2;
expBrk  = isBrk & onset >= 2;
expNB   = cls == 0 & trackT >= opts.minTrackT;
retain  = (expBrk & tPreT >= opts.minLeadT) | expNB;
expList = find(expBrk | expNB);
fprintf('\n%d tracked waves; exported: %d breakers with at least one scan before onset (%d retained: %d spilling, %d plunging) and %d non-breaking controls\n', ...
    nW, sum(expBrk), sum(expBrk & retain), sum(expBrk & retain & cls == 2), sum(expBrk & retain & cls == 3), sum(expNB));

%% ---- 2. Still-water levels ---------------------------------------------------
swlTide = nan(nW, 1);  swlLidar = nan(nW, 1);  swlRecord = nan(nW, 1);
if exist(fileFigs, 'file') == 2
    ws = warning('off', 'all');                    % function handles saved by the authors
    G = load(fileFigs, 'WaveCond');
    warning(ws);
    if isfield(G, 'WaveCond')
        wc = G.WaveCond(1);
        if all(isfield(wc, {'tidegauge_time', 'tidegauge_zMSL'}))
            tg = toEpoch(wc.tidegauge_time);
            zg = double(wc.tidegauge_zMSL(:));
            ok = isfinite(tg) & isfinite(zg);
            tgo = tg(ok);
            zgo = zg(ok);
            [tgs, iu] = unique(tgo);
            if numel(tgs) > 1, swlTide = interp1(tgs, zgo(iu), t0, 'linear', NaN); end
        end
        if all(isfield(wc, {'lidar_time', 'lidar_zMSL'}))
            tl = toEpoch(wc.lidar_time);
            zl = double(wc.lidar_zMSL(:));
            for k = 1:nW
                [dt, i] = min(abs(tl - t0(k)));
                if ~isempty(dt) && dt <= 1800, swlLidar(k) = zl(i); end
            end
        end
    end
end
if all(isfield(S, {'tstartcat', 'zMSLcat'}))
    ts = toEpoch(S.tstartcat);
    zr = double(S.zMSLcat(:));
    for k = 1:nW
        r = find(ts <= t0(k) + 1, 1, 'last');
        if ~isempty(r) && r <= numel(zr) && t0(k) - ts(r) <= 3600, swlRecord(k) = zr(r); end
    end
end

%% ---- 3. Profiles ----------------------------------------------------------------
fprintf('\nLoading the profiles (about 3 GB) ...\n');
tic;
P = load(fileWaves, 'xprofilesCat', 'zprofilesCat');
fprintf('  loaded in %.0f s\n', toc);

%% ---- 4. Export --------------------------------------------------------------------
template = struct('id', NaN, 'type', NaN, 'typeName', '', 'retained', false, 'day', '', ...
    'onset', NaN, 'codeLenDiff', 0, 'nScans', NaN, 'tEpoch0', NaN, 't', [], 'x', [], 'dx', NaN, ...
    'eta', [], 'gapFill', [], 'hMSL', [], 'xc', [], 'zc', [], 'H', [], 'ztr', [], 'C', [], ...
    'T', NaN, 'tPre', NaN, 'tPre_T', NaN, 'track_T', NaN, 'hOnset', NaN, 'HOnset', NaN, ...
    'gammaOnset', NaN, 'COnset', NaN, 'xOnset', NaN, 'gapFracCrest', NaN, 'gapFracAll', NaN, ...
    'swlTide', NaN, 'swlLidar', NaN, 'swlRecord', NaN, 'etaMean', NaN, 'gridDev', NaN);
nE = numel(expList);
waves = repmat(template, nE, 1);
fprintf('\nExporting %d waves ...\n', nE);
tic;
for e = 1:nE
    k  = expList(e);
    t  = toEpoch(cellOf(S.tcat, k));
    nT = numel(t);
    xc = fieldVec(S, 'xcat', k);
    tr = fieldVec(S, 'ztrprecat', k);
    [xg, eta, dev] = rebuildWave(cellOf(P.xprofilesCat, k), cellOf(P.zprofilesCat, k), xc, tr, nT);
    nX  = numel(xg);
    dx  = medf(diff(xg));
    gap = gapRuns(eta, dx, opts.gapRunMin, opts.straightTol);
    rc  = crestRows(fieldVec(S, 'ixcat', k), nX, nT);
    w   = max(1, round(opts.crestWindow / dx));
    if expBrk(k), sc = 1:onset(k)-1; else, sc = 1:nT; end
    [fCrest, fAll] = gapShares(eta, gap, rc, w, sc);

    fin  = any(isfinite(eta), 2);
    if any(fin), keep = find(fin, 1):opts.xStride:find(fin, 1, 'last'); else, keep = []; end
    hr   = depthRow(S, k, nW);
    Hk   = fieldVec(S, 'Hcat', k);
    Cv   = nan(nT, 1);
    for j = 1:nT, Cv(j) = crestSpeed(t, xc, j); end

    v = template;
    v.id = k;  v.type = cls(k);  v.typeName = typeName(cls(k));  v.retained = retain(k);
    v.day = datestr(t(1) / 86400 + datenum(1970, 1, 1), 'yyyy-mm-dd');
    v.onset = onset(k);  v.codeLenDiff = lenDiff(k);  v.nScans = nT;
    v.tEpoch0 = t(1);  v.t = t - t(1);
    v.x = xg(keep);  v.dx = dx * opts.xStride;
    v.eta = single(eta(keep, :));
    v.gapFill = gap(keep, :);
    if numel(hr) == nX, v.hMSL = single(hr(keep)); end
    v.xc = xc;  v.zc = fieldVec(S, 'zcat', k);  v.H = Hk;  v.ztr = tr;  v.C = Cv;
    v.T = Tw(k);  v.tPre = tPre(k);  v.tPre_T = tPreT(k);  v.track_T = trackT(k);
    if ~isnan(onset(k))
        io = onset(k);
        if ~isnan(rc(io)) && numel(hr) == nX, v.hOnset = hr(rc(io)); end
        v.HOnset = at(Hk, io);
        v.gammaOnset = v.HOnset / v.hOnset;
        v.COnset = Cv(io);
        v.xOnset = at(xc, io);
    end
    v.gapFracCrest = fCrest;  v.gapFracAll = fAll;
    v.swlTide = swlTide(k);  v.swlLidar = swlLidar(k);  v.swlRecord = swlRecord(k);
    v.etaMean = meanf(eta(:));
    v.gridDev = dev;
    waves(e) = v;
    if mod(e, 200) == 0, fprintf('  %d of %d waves (%.0f s)\n', e, nE, toc); end
end

%% ---- 5. Check against the raw lidar of the example record ---------------------------
if opts.checkExample && exist(fileFigs, 'file') == 2
    ws = warning('off', 'all');                    % function handles saved by the authors
    try
        G = load(fileFigs, 'Tracking');
        warning(ws);
        checkExample(G.Tracking, S, P, cls, onset, nW, opts);
        clear G
    catch err
        warning(ws);
        fprintf('\nThe example record could not be checked: %s\n', err.message);
    end
end
clear P

%% ---- 6. Save -------------------------------------------------------------------------
meta = struct();
meta.description = ['Duck 2016 lidar waves (Carini et al., data CC BY, http://hdl.handle.net/1773/46586) ', ...
    'exported for the PINN field test of the breaking-inception parameter B'];
meta.created  = datestr(now, 'yyyy-mm-dd HH:MM');
meta.source   = fileWaves;
meta.units    = ['x, dx, eta, hMSL, xc, zc, H, ztr, hOnset, xOnset, swl*, etaMean in m; t, T, tPre in s ', ...
    '(t from the first scan, tEpoch0 in s since 1970-01-01 UTC); C, COnset in m/s'];
meta.arrays   = 'eta and gapFill are nX-by-nT: one row per grid point x, one column per scan t';
meta.eta      = ['free surface on the fixed FRF grid x, in the datum of the tracked crests ', ...
    '(zprofilesCat + ztrprecat); NaN where the profile has no data'];
meta.hMSL     = 'water depth below mean sea level along x (hMSLcat)';
meta.gapFill  = sprintf(['true where the gridded profile is a straight line over at least %.2f m, ', ...
    'probably interpolated across a gap in the lidar returns'], opts.gapRunMin);
meta.onset    = 'first scan labelled spilling (2) or plunging (3) in manTypecat; NaN for non-breaking waves';
meta.retained = sprintf(['breakers with at least %.2f T of record before onset; non-breaking waves ', ...
    'tracked for at least %.2f T'], opts.minLeadT, opts.minTrackT);
meta.C        = 'crest speed from a straight-line fit to five consecutive positions of the tracked crest';
meta.swl      = ['still-water level candidates: swlTide (tide gauge), swlLidar (lidar mean level), ', ...
    'swlRecord (zMSLcat of the record), etaMean (mean of eta over the wave window)'];
meta.opts     = opts;

idx = struct();
idx.id = [waves.id]';  idx.type = [waves.type]';  idx.retained = [waves.retained]';
idx.onset = [waves.onset]';  idx.nScans = [waves.nScans]';  idx.tPre_T = [waves.tPre_T]';
idx.track_T = [waves.track_T]';  idx.hOnset = [waves.hOnset]';  idx.gammaOnset = [waves.gammaOnset]';
idx.COnset = [waves.COnset]';  idx.gapFracCrest = [waves.gapFracCrest]';

save(outFile, 'waves', 'meta', 'idx', '-v7');
writeIndex(csvFile, waves);
fileInfo = dir(outFile);

%% ---- 7. Summary -------------------------------------------------------------------------
ty = [waves.type]';  re = [waves.retained]';
fprintf('\n================ SUMMARY ================\n');
fprintf('Exported %d waves to %s (%.0f MB):\n', nE, outFile, fileInfo.bytes / 2^20);
fprintf('  spilling %d (retained %d), plunging %d (retained %d), non-breaking %d\n', ...
    sum(ty == 2), sum(ty == 2 & re), sum(ty == 3), sum(ty == 3 & re), sum(ty == 0));
fprintf('Grid: every %d points of the 1-cm grid; largest change of the rebuilt FRF grid between scans: %.2g m\n', ...
    opts.xStride, max([waves.gridDev]));
for c = [2 3 0]
    sel = ty == c & re;
    q = pctf([waves(sel).gapFracCrest], [50 90]);
    fprintf('  %-12s share of points within %.0f m of the crest%s flagged as gap fills: median %.3f, 90th pct %.3f\n', ...
        typeName(c), opts.crestWindow, ternary(c > 0, ' before onset', ''), q(1), q(2));
end
fprintf('Still-water level (m above MSL), medians over the exported waves:\n');
fprintf('  tide gauge %.2f, lidar %.2f, record (zMSLcat) %.2f, mean of the gridded surface %.2f\n', ...
    medf([waves.swlTide]), medf([waves.swlLidar]), medf([waves.swlRecord]), medf([waves.etaMean]));
fprintf('  differences from the tide gauge: lidar %.2f, record %.2f, surface mean %.2f\n', ...
    medf([waves.swlLidar] - [waves.swlTide]), medf([waves.swlRecord] - [waves.swlTide]), ...
    medf([waves.etaMean] - [waves.swlTide]));
dayList = unique({waves.day});
for d = 1:numel(dayList)
    sel = strcmp({waves.day}, dayList{d});
    fprintf('  %s: %d waves; tide gauge %.2f, record %.2f, surface mean %.2f\n', dayList{d}, sum(sel), ...
        medf([waves(sel).swlTide]), medf([waves(sel).swlRecord]), medf([waves(sel).etaMean]));
end
fprintf('Index written to %s\n', csvFile);

diary off;
fprintf('\nLog written to %s\n', logFile);


%% ======================== local functions =====================================

function nm = typeName(c)
% Name of a wave type code (-1 = in no class list).
    allNames = {'unclassified', 'nonbreaking', 'steady state', 'spilling', 'plunging'};
    nm = allNames{c + 2};
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
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

function m = medf(v)
% Median of the finite values (NaN if there are none).
    v = double(v(isfinite(v)));
    if isempty(v), m = NaN; else, m = median(v(:)); end
end

function m = maxf(v)
% Largest finite value (NaN if there are none).
    v = double(v(isfinite(v)));
    if isempty(v), m = NaN; else, m = max(v(:)); end
end

function m = meanf(v)
% Mean of the finite values (NaN if there are none).
    v = double(v(isfinite(v)));
    if isempty(v), m = NaN; else, m = mean(v(:)); end
end

function q = pctf(x, p)
% Percentiles P of the finite values of X (no toolbox needed).
    x = sort(double(x(isfinite(x))));
    if isempty(x), q = nan(size(p)); return; end
    if numel(x) == 1, q = repmat(x, size(p)); return; end
    q = interp1(linspace(0, 100, numel(x)), x, p);
end

function t = toEpoch(t)
% Time in seconds since 1970-01-01 UTC, from epoch seconds, milliseconds or datenum days.
    t = double(t(:));
    tm = medf(t);
    if tm > 1e11
        t = t / 1000;
    elseif tm > 7e5 && tm < 1e6
        t = (t - 719529) * 86400;                    % 719529 = datenum(1970, 1, 1)
    end
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

function [xg, eta, dev] = rebuildWave(X, Z, xc, ztr, nT)
% Fixed FRF grid xg and free surface eta (nX-by-nT) of one wave:
% x = xprofile + xcat (crest-relative grid), eta = zprofile + ztrprecat (trough-referenced).
% dev is the largest change of the rebuilt grid between scans; scans whose grid differs
% from that of the first scan are interpolated onto it.
    x1  = getScan(X, 1, nT).' + at(xc, 1);
    xg  = x1;
    nX  = numel(xg);
    eta = nan(nX, nT);
    dev = 0;
    for j = 1:nT
        xj = getScan(X, j, nT).' + at(xc, j);
        zj = getScan(Z, j, nT).' + at(ztr, j);
        d  = max(abs(xj - xg));
        dev = max(dev, d);
        if d < 1e-6
            eta(:, j) = zj;
        else
            ok = isfinite(xj) & isfinite(zj);
            if sum(ok) > 1, eta(:, j) = interp1(xj(ok), zj(ok), xg, 'linear', NaN); end
        end
    end
end

function G = gapRuns(E, dx, minLen, tol)
% True inside stretches where a profile is a straight line over at least minLen
% (linear interpolation across a gap in the lidar returns). E is nX-by-nT; NaN gives false.
    [nX, nT] = size(E);
    G = false(nX, nT);
    nMin = max(3, ceil(minLen / dx - 1e-9) + 1);   % grid points spanning minLen
    for j = 1:nT
        z  = E(:, j);
        d2 = abs(z(1:end-2) - 2 * z(2:end-1) + z(3:end));
        s  = [false; d2 < tol; false];             % interior points between collinear neighbours
        e  = diff([false; s; false]);
        b  = find(e == 1);
        f  = find(e == -1) - 1;
        for r = 1:numel(b)
            if f(r) - b(r) + 3 >= nMin             % the straight stretch runs from b-1 to f+1
                G(max(1, b(r) - 1):min(nX, f(r) + 1), j) = true;
            end
        end
    end
end

function rc = crestRows(ix, nX, nT)
% Crest row on the grid for each scan from ixcat (NaN where missing or out of range).
    rc = nan(nT, 1);
    n = min(nT, numel(ix));
    rc(1:n) = ix(1:n);
    rc(~(rc >= 1 & rc <= nX & rc == round(rc))) = NaN;
end

function [fCrest, fAll] = gapShares(E, G, rc, w, sc)
% Share of the finite points flagged as gap fills within w points of the crest, and over
% the whole profile, for the scans sc.
    nC = 0; gC = 0; nA = 0; gA = 0;
    nX = size(E, 1);
    for j = sc(:)'
        ok = isfinite(E(:, j));
        nA = nA + sum(ok);
        gA = gA + sum(ok & G(:, j));
        if ~isnan(rc(j))
            rows = max(1, rc(j) - w):min(nX, rc(j) + w);
            nC = nC + sum(ok(rows));
            gC = gC + sum(ok(rows) & G(rows, j));
        end
    end
    fCrest = gC / max(nC, 1);
    fAll   = gA / max(nA, 1);
    if nC == 0, fCrest = NaN; end
    if nA == 0, fAll = NaN; end
end

function h = depthRow(S, k, nW)
% Depth below MSL along the grid for wave k (hMSLcat holds one row per wave).
    h = [];
    if ~isfield(S, 'hMSLcat'), return; end
    D = S.hMSLcat;
    if iscell(D)
        h = double(D{k});
    elseif size(D, 1) == nW
        h = double(D(k, :));
    elseif size(D, 2) == nW
        h = double(D(:, k));
    end
    h = h(:);
end

function C = crestSpeed(t, xc, i)
% Crest speed |dx/dt| from a straight-line fit to the tracked crest positions,
% 5 scans around scan i.
    C = NaN;
    n = min(numel(t), numel(xc));
    if n < 3 || i > n, return; end
    idx = max(1, i - 2):min(n, i + 2);
    idx = idx(isfinite(xc(idx)) & isfinite(t(idx)));
    if numel(idx) < 3, return; end
    p = polyfit(t(idx) - t(idx(1)), xc(idx), 1);
    C = abs(p(1));
end

function [d, gapLen] = distToRaw(xs, xr)
% Distance from each grid point xs to the nearest raw lidar return xr, and the length of
% the interval between the raw returns on either side of it (0 at a return, NaN outside).
    xr = unique(double(xr(isfinite(xr))));
    if numel(xr) < 2, d = inf(size(xs)); gapLen = nan(size(xs)); return; end
    nn = interp1(xr, xr, xs, 'nearest', 'extrap');
    d = abs(xs - nn);
    gapLen = interp1(xr, xr, xs, 'next') - interp1(xr, xr, xs, 'previous');
end

function checkExample(Tr, S, P, cls, onset, nW, opts)
% Compares gridded surfaces with the one record of the second file that keeps the lidar
% returns (Tracking.x, Tracking.z for every scan):
%   (A) how that record was gridded (Tracking.zs) and how its gaps were filled;
%   (B) gaps near the crests tracked in that record (Tracking.trackspace);
%   (C) the waves of the per-wave file recorded in it, against the same returns.
    fprintf('\nExample record in the second file (Tracking):\n');
    for f = {'tt', 'target', 'z_MSL'}
        if isfield(Tr, f{1}) && isnumeric(Tr.(f{1})) && isscalar(Tr.(f{1}))
            fprintf('  %-7s %g\n', f{1}, Tr.(f{1}));
        end
    end
    if ~all(isfield(Tr, {'xs', 'zs', 't'}))
        fprintf('  xs, zs or t missing: no check.\n');
        return
    end
    xs  = double(Tr.xs(:));
    zs  = Tr.zs;
    tr  = toEpoch(Tr.t);
    nS  = numel(tr);
    nXs = numel(xs);
    win = max(1, round(opts.crestWindow / medf(diff(xs))));
    fprintf('  grid xs: %d points, %.3f to %.3f m (step %.4f); zs %s; %d scans from %s to %s UTC\n', ...
        nXs, xs(1), xs(end), medf(diff(xs)), mat2str(size(zs)), nS, ...
        datestr(tr(1) / 86400 + datenum(1970, 1, 1), 'yyyy-mm-dd HH:MM:SS'), ...
        datestr(tr(end) / 86400 + datenum(1970, 1, 1), 'HH:MM:SS'));
    hasRaw = all(isfield(Tr, {'x', 'z'})) && iscell(Tr.x) && iscell(Tr.z) && numel(Tr.x) == nS;
    if hasRaw
        nr = cellfun(@numel, Tr.x);
        if all(nr(1:min(20, end)) == nXs)
            fprintf('  x, z have as many values per scan as the grid: they are not raw returns.\n');
            hasRaw = false;
        else
            fprintf('  lidar returns: %d to %d per scan (median %d)\n', min(nr), max(nr), round(medf(nr)));
        end
    else
        fprintf('  no per-scan lidar returns (x, z).\n');
    end
    ts = [];
    if isfield(Tr, 'trackspace') && isstruct(Tr.trackspace), ts = Tr.trackspace; end
    if ~isempty(ts) && isfield(ts, 'gappyInterp') && iscell(ts.gappyInterp)
        g = cellfun(@(v) double(v(:)'), ts.gappyInterp, 'UniformOutput', false);
        gv = [g{:}];
        gv = gv(isfinite(gv));
        u = unique(gv);
        fprintf('  trackspace.gappyInterp: %d tracked waves, %d scans; values %s; share non-zero %.3f\n', ...
            numel(g), numel(gv), mat2str(u(1:min(8, end)), 4), mean(gv ~= 0));
    end

    % (A) how the record was gridded and how its gaps were filled
    if hasRaw
        nIn = 0; nNaN = 0; nFin = 0; nUns = 0; nGap = 0; nGapFin = 0;
        dBin = []; dLin = []; dGap = [];
        for i = 1:10:nS
            [xr, zr] = rawScan(Tr, i);
            if numel(xr) < 2, continue; end
            zc = double(zs(:, i));
            [d, gapLen] = distToRaw(xs, xr);
            inSpan = xs >= xr(1) & xs <= xr(end);
            fin = isfinite(zc) & inSpan;
            gp  = inSpan & gapLen >= opts.gapRunMin;
            zb  = binMean(xs, xr, zr);
            zl  = interp1(xr, zr, xs, 'linear', NaN);
            nIn = nIn + sum(inSpan);           nNaN = nNaN + sum(inSpan & ~isfinite(zc));
            nFin = nFin + sum(fin);            nUns = nUns + sum(fin & d > opts.dRaw);
            nGap = nGap + sum(gp);             nGapFin = nGapFin + sum(gp & isfinite(zc));
            ok = fin & isfinite(zb);           dBin = [dBin; zc(ok) - zb(ok)]; %#ok<AGROW>
            ok = fin & isfinite(zl) & ~gp;     dLin = [dLin; zc(ok) - zl(ok)]; %#ok<AGROW>
            ok = gp & isfinite(zc) & isfinite(zl);  dGap = [dGap; zc(ok) - zl(ok)]; %#ok<AGROW>
        end
        fprintf('  (A) gridding of the record (every 10th scan), within the span of the returns:\n');
        fprintf('      grid points with no value in zs: %.3f; finite points with no return within %.0f cm: %.3f\n', ...
            nNaN / max(nIn, 1), 100 * opts.dRaw, nUns / max(nFin, 1));
        fprintf('      grid points in gaps of at least %.0f cm between returns: %.3f, of which %.3f have a value (filled)\n', ...
            100 * opts.gapRunMin, nGap / max(nIn, 1), nGapFin / max(nGap, 1));
        fprintf('      zs minus the mean of the returns in its 1-cm cell:      %s\n', offMad(dBin));
        fprintf('      zs minus a straight line between returns, outside gaps: %s\n', offMad(dLin));
        fprintf('      zs minus a straight line across gaps of %.0f cm or more: %s\n', 100 * opts.gapRunMin, offMad(dGap));
    end

    % (B) the crests tracked in the record, scans before onset
    if hasRaw && ~isempty(ts) && all(isfield(ts, {'ixpk', 'itpk'})) && iscell(ts.ixpk)
        c = zeros(2, 7);                       % rows: gappyInterp 0 / non-zero; columns: see below
        nWav = 0;
        for w = 1:numel(ts.ixpk)
            ix = double(ts.ixpk{w}(:));
            it = double(ts.itpk{w}(:));
            n  = min(numel(ix), numel(it));
            last = n;
            if isfield(ts, 'ManType') && w <= numel(ts.ManType)
                code = double(ts.ManType{w}(:));
                i2 = find(code >= 2, 1);
                if numel(code) == n && ~isempty(i2), last = i2 - 1; end
            end
            gpw = [];
            if isfield(ts, 'gappyInterp') && w <= numel(ts.gappyInterp), gpw = double(ts.gappyInterp{w}(:)); end
            used = false;
            for j = 1:last
                i = it(j);
                r = ix(j);
                if ~(i >= 1 && i <= nS && r >= 1 && r <= nXs && i == round(i) && r == round(r)), continue; end
                [xr, zr] = rawScan(Tr, i); %#ok<ASGLU>
                if numel(xr) < 2, continue; end
                rows = max(1, r - win):min(nXs, r + win);
                [d, gapLen] = distToRaw(xs(rows), xr);
                inSpan = xs(rows) >= xr(1) & xs(rows) <= xr(end);
                fin = isfinite(double(zs(rows, i))) & inSpan;
                gp  = inSpan & gapLen >= opts.gapRunMin;
                q = 1 + (j <= numel(gpw) && gpw(j) ~= 0);
                % in span, finite, finite without a return within dRaw, in gaps, gaps with a value,
                % scans, scans with a return within dRaw of the crest point
                c(q, :) = c(q, :) + [sum(inSpan), sum(fin), sum(fin & d > opts.dRaw), sum(gp), ...
                    sum(gp & fin), 1, double(distToRaw(xs(r), xr) <= opts.dRaw)];
                used = true;
            end
            nWav = nWav + used;
        end
        tot = sum(c, 1);
        fprintf('  (B) %d crests tracked in the record, %d scans before onset; within %.0f m of the crest:\n', ...
            nWav, tot(6), opts.crestWindow);
        fprintf('      grid points in gaps of at least %.0f cm: %.3f (%.3f of them filled); finite points with no return within %.0f cm: %.3f\n', ...
            100 * opts.gapRunMin, tot(4) / max(tot(1), 1), tot(5) / max(tot(4), 1), 100 * opts.dRaw, tot(3) / max(tot(2), 1));
        fprintf('      a return within %.0f cm of the crest point itself: %.3f of the scans\n', 100 * opts.dRaw, tot(7) / max(tot(6), 1));
        if c(2, 6) > 0
            fprintf('      scans with gappyInterp = 0: %d, share in gaps %.3f; scans with gappyInterp ~= 0: %d, share in gaps %.3f\n', ...
                c(1, 6), c(1, 4) / max(c(1, 1), 1), c(2, 6), c(2, 4) / max(c(2, 1), 1));
        end
    end

    % (C) waves of the per-wave file recorded in this record, matched by time
    members = [];
    maps = {};
    [tu, iu] = unique(tr);                     % sorted scan times, for the nearest-scan lookup
    for k = 1:nW
        tk = toEpoch(cellOf(S.tcat, k));
        if isempty(tk) || tk(1) < tu(1) - 0.5 || tk(end) > tu(end) + 0.5, continue; end
        if numel(tu) > 1
            i = iu(interp1(tu, (1:numel(tu))', tk, 'nearest', 'extrap'));
        else
            i = repmat(iu(1), size(tk));
        end
        if max(abs(tr(i) - tk)) < 0.05
            members(end+1) = k; %#ok<AGROW>
            maps{end+1} = i(:); %#ok<AGROW>
        end
    end
    fprintf('  (C) waves of the per-wave file recorded in this record (matched by time): %d (%s)\n', numel(members), ...
        strjoin(arrayfun(@(cc) sprintf('%s %d', typeName(cc), sum(cls(members) == cc)), -1:3, ...
        'UniformOutput', false), ', '));
    if isempty(members), return; end
    nAgree = 0; nScan = 0;
    dZs = []; dCrest = []; eBin = []; eLin = []; eGap = [];
    cnt = zeros(1, 5);                         % near crest: in span, finite, in gaps, gaps with a value, finite without a return within dRaw
    for m = 1:numel(members)
        k  = members(m);
        im = maps{m};
        tk = toEpoch(cellOf(S.tcat, k));
        nT = numel(tk);
        xc = fieldVec(S, 'xcat', k);
        [xg, eta] = rebuildWave(cellOf(P.xprofilesCat, k), cellOf(P.zprofilesCat, k), xc, ...
            fieldVec(S, 'ztrprecat', k), nT);
        it = fieldVec(S, 'itcat', k);
        if numel(it) == nT, nAgree = nAgree + sum(it == im); end
        nScan = nScan + nT;
        ig  = round((xg - xs(1)) / medf(diff(xs))) + 1;
        okg = ig >= 1 & ig <= nXs;
        okg(okg) = abs(xs(ig(okg)) - xg(okg)) < medf(diff(xg)) / 2;
        rc  = crestRows(fieldVec(S, 'ixcat', k), numel(xg), nT);
        wg  = max(1, round(opts.crestWindow / medf(diff(xg))));
        zc  = fieldVec(S, 'zcat', k);
        if cls(k) >= 2 && ~isnan(onset(k)), sc = 1:onset(k)-1; else, sc = 1:nT; end
        for j = sc
            col = nan(numel(xg), 1);
            col(okg) = double(zs(ig(okg), im(j)));
            dZs = [dZs; eta(:, j) - col]; %#ok<AGROW>
            if ~isnan(rc(j)), dCrest(end+1) = col(rc(j)) - at(zc, j); end %#ok<AGROW>
            if ~hasRaw || isnan(rc(j)), continue; end
            [xr, zr] = rawScan(Tr, im(j));
            if numel(xr) < 2, continue; end
            rows = max(1, rc(j) - wg):min(numel(xg), rc(j) + wg);
            [d, gapLen] = distToRaw(xg(rows), xr);
            e  = eta(rows, j);
            inSpan = xg(rows) >= xr(1) & xg(rows) <= xr(end);
            gp  = inSpan & gapLen >= opts.gapRunMin;
            fin = isfinite(e) & inSpan;
            cnt = cnt + [sum(inSpan), sum(fin), sum(gp), sum(gp & fin), sum(fin & d > opts.dRaw)];
            zb = binMean(xg(rows), xr, zr);
            zl = interp1(xr, zr, xg(rows), 'linear', NaN);
            ok = fin & isfinite(zb);            eBin = [eBin; e(ok) - zb(ok)]; %#ok<AGROW>
            ok = fin & isfinite(zl) & ~gp;      eLin = [eLin; e(ok) - zl(ok)]; %#ok<AGROW>
            ok = fin & isfinite(zl) & gp;       eGap = [eGap; e(ok) - zl(ok)]; %#ok<AGROW>
        end
    end
    fprintf('      itcat equals the matched scan of the record in %.3f of the scans\n', nAgree / max(nScan, 1));
    fprintf('      rebuilt surface minus zs at the same scans:   %s\n', offMad(dZs));
    fprintf('      zs at the tracked crest minus zcat:            %s\n', offMad(dCrest));
    if hasRaw
        fprintf('      within %.0f m of the crest, scans before onset: %.3f of the grid points in gaps of at least %.0f cm (%.3f of them filled); finite points with no return within %.0f cm: %.3f\n', ...
            opts.crestWindow, cnt(3) / max(cnt(1), 1), 100 * opts.gapRunMin, cnt(4) / max(cnt(3), 1), ...
            100 * opts.dRaw, cnt(5) / max(cnt(2), 1));
        fprintf('      rebuilt surface minus the mean of the returns in its 1-cm cell: %s\n', offMad(eBin));
        fprintf('      rebuilt surface minus a straight line between returns, outside gaps: %s\n', offMad(eLin));
        fprintf('      rebuilt surface minus a straight line across gaps:                   %s\n', offMad(eGap));
    end
end

function s = offMad(d)
% 'median difference (offset) and median absolute deviation about it' for printing.
    d = d(isfinite(d));
    if isempty(d), s = 'no points'; return; end
    o = median(d);
    s = sprintf('offset %+.4f m, spread %.4f m (%d points)', o, median(abs(d - o)), numel(d));
end

function [xr, zr] = rawScan(Tr, i)
% Lidar returns of scan i: sorted unique x, mean z at repeated x, as columns.
    xr = double(Tr.x{i});
    zr = double(Tr.z{i});
    xr = xr(:);
    zr = zr(:);
    n  = min(numel(xr), numel(zr));
    xr = xr(1:n);
    zr = zr(1:n);
    ok = isfinite(xr) & isfinite(zr);
    if sum(ok) < 2, xr = xr(ok); zr = zr(ok); return; end
    [xr, ~, ic] = unique(xr(ok));
    zr = accumarray(ic, zr(ok), [], @mean);
end

function zb = binMean(xs, xr, zr)
% Mean of the returns in the 1-cm cell of each grid point (NaN where there is none).
    dx = medf(diff(xs));
    b  = round((xr - xs(1)) / dx) + 1;
    ok = b >= 1 & b <= numel(xs);
    zb = accumarray(b(ok), zr(ok), [numel(xs) 1], @mean, NaN);
end

function writeIndex(f, W)
% One line per exported wave, for a quick look (pandas.read_csv).
    cols = {'id', 'type', 'typeName', 'retained', 'day', 'onset', 'codeLenDiff', 'nScans', ...
            'tEpoch0', 'T', 'tPre', 'tPre_T', 'track_T', 'hOnset', 'HOnset', 'gammaOnset', ...
            'COnset', 'xOnset', 'gapFracCrest', 'gapFracAll', 'swlTide', 'swlLidar', ...
            'swlRecord', 'etaMean', 'gridDev'};
    fid = fopen(f, 'w');
    if fid < 0, error('writeIndex:open', 'Cannot write %s', f); end
    fprintf(fid, '%s\n', strjoin(cols, ','));
    for i = 1:numel(W)
        for c = 1:numel(cols)
            v = W(i).(cols{c});
            if ischar(v)
                fprintf(fid, '%s', v);
            elseif islogical(v)
                fprintf(fid, '%d', v);
            else
                fprintf(fid, '%.10g', v);
            end
            if c < numel(cols), fprintf(fid, ','); end
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
end
