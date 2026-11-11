%% carini_gap_study.m   (version 1, 10 Nov 2026)
% How much of the gridded Duck 2016 lidar surface near the crests is measured, and how much is
% filled across gaps between lidar returns. Uses the one record of SurfZoneWavesAtOnsetBreaking.mat
% that keeps the returns (Tracking: gridded stack zs on xs, returns x and z for every scan, and 46
% tracked crests). Run carini_export_pinn.m first: step 5 adds a mask to its output.
%
% What it does
%   1. Reads the example record and lists the crests tracked in it (scans before onset).
%   2. Finds how the record was gridded: compares zs with candidate griddings of the returns
%      (straight lines or pchip through the returns; 1-cm cell means or medians, with gaps filled
%      by straight lines, pchip or a cubic that makes the fourth difference vanish).
%   3. Tests detectors of filled stretches in a gridded profile (zero second or fourth
%      differences, low roughness) against the true gaps between returns, and picks the best.
%   4. Measures where returns fall around the crests, the lengths of the gaps (for the lidar
%      sampling of the synthetic twin) and the error of the filled surface, by removing returns
%      over 0.1-1.6 m and gridding again with the best candidate.
%   5. If a detector works on the record (F1 >= opts.minF1 near the crests), flags the likely
%      filled points of every exported wave: carini_pinn_mask.mat, which carini_pinn_load.py
%      reads automatically.
%   6. Figure carini_gap_study: one crest with its returns, the share of the surface in gaps
%      around the crest, and the error of the fill against gap length.
% Outputs: carini_gap_study_log.txt, carini_lidar_pattern.mat, carini_pinn_mask.mat (if a
% detector works), carini_gap_study.png/.pdf.

clear; clc; close all;

%% ---- Settings ------------------------------------------------------------
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
fileWaves = fullfile(here, 'lidarIR_sharedData.mat');
fileFigs  = fullfile(here, 'SurfZoneWavesAtOnsetBreaking.mat');
fileExp   = fullfile(here, 'carini_pinn_input.mat');
logFile   = fullfile(here, 'carini_gap_study_log.txt');

opts.gapMin      = 0.10;   % [m] a gap is a stretch of at least this length between returns
opts.crestWindow = 2.0;    % [m] half-width of the crest region
opts.range       = 8.0;    % [m] half-width of the region around the crest in the pattern
opts.bin         = 0.05;   % [m] bin of the pattern
opts.sampleEvery = 10;     % use every 10th scan of the record for the whole-profile statistics
opts.removeL     = [0.1 0.2 0.4 0.8 1.6];   % [m] lengths of the returns removed in step 4
opts.nTrials     = 80;     % trials per length and position in step 4
opts.minF1       = 0.7;    % a detector is used for the masks only if F1 near the crests >= this
opts.makeFigures = true;

assert(exist(fileFigs, 'file') == 2, 'Not found: %s', fileFigs);
if exist(logFile, 'file') == 2, delete(logFile); end
diary(logFile);
fprintf('carini_gap_study.m, version 1, run %s\n', datestr(now, 'yyyy-mm-dd HH:MM'));
if exist('rng', 'file') || exist('rng', 'builtin'), rng(1); else, rand('seed', 1); end %#ok<RAND>

%% ---- 1. Example record and its tracked crests --------------------------------
ws = warning('off', 'all');                        % function handles saved by the authors
G = load(fileFigs, 'Tracking');
warning(ws);
Tr = G.Tracking;
clear G
xs  = double(Tr.xs(:));
zs  = Tr.zs;
tr  = toEpoch(Tr.t);
nS  = numel(tr);
nXs = numel(xs);
dx  = medf(diff(xs));
assert(all(isfield(Tr, {'x', 'z'})) && iscell(Tr.x) && numel(Tr.x) == nS, 'The record keeps no lidar returns.');
nRet = cellfun(@numel, Tr.x);
fprintf('\nRecord %s to %s UTC: %d scans, grid %.2f to %.2f m at %.3f m; returns per scan %d to %d (median %d)\n', ...
    datestr(tr(1) / 86400 + datenum(1970, 1, 1), 'yyyy-mm-dd HH:MM:SS'), ...
    datestr(tr(end) / 86400 + datenum(1970, 1, 1), 'HH:MM:SS'), nS, xs(1), xs(end), dx, ...
    min(nRet), max(nRet), round(medf(nRet)));

ts = Tr.trackspace;
crest = zeros(0, 4);                               % record scan, crest row, crest number, scan of the crest
nOnset = 0;
for w = 1:numel(ts.ixpk)
    ix = double(ts.ixpk{w}(:));
    it = double(ts.itpk{w}(:));
    n  = min(numel(ix), numel(it));
    last = n;
    if isfield(ts, 'ManType') && w <= numel(ts.ManType)
        code = double(ts.ManType{w}(:));
        i2 = find(code >= 2, 1);
        if ~isempty(i2) && i2 <= n, last = i2 - 1; nOnset = nOnset + 1; end
    end
    for j = 1:last
        if it(j) >= 1 && it(j) <= nS && ix(j) >= 1 && ix(j) <= nXs && it(j) == round(it(j)) && ix(j) == round(ix(j))
            crest(end+1, :) = [it(j), ix(j), w, j]; %#ok<AGROW>
        end
    end
end
fprintf('Tracked crests: %d (%d with an observed onset); %d scans before onset are used\n', ...
    numel(ts.ixpk), nOnset, size(crest, 1));

%% ---- 2. How the record was gridded ----------------------------------------------
cands = {'returns, straight lines', 'returns, pchip', 'cell means, straight-line fill', ...
         'cell means, pchip fill', 'cell means, cubic fill', 'left-aligned cell means, cubic fill', ...
         'cell medians, cubic fill', 'smoothed cell means, cubic fill'};
nC = numel(cands);
dMeas = cell(1, nC);
dGap  = cell(1, nC);
scansA = 1:opts.sampleEvery:nS;
for i = scansA
    [xr, zr] = rawScan(Tr, i);
    if numel(xr) < 3, continue; end
    zc = double(zs(:, i));
    [~, gapLen, inCell] = rawSupport(xs, xr, dx);
    use = isfinite(zc) & xs >= xr(1) & xs <= xr(end);
    gp  = gapLen >= opts.gapMin;
    for c = 1:nC
        zg = gridReturns(cands{c}, xs, xr, zr);
        ok = use & isfinite(zg);
        dMeas{c} = [dMeas{c}; zg(ok & inCell) - zc(ok & inCell)];
        dGap{c}  = [dGap{c}; zg(ok & gp) - zc(ok & gp)];
    end
end
fprintf('\nGridding of the record (every %d scans): |candidate - zs| at grid points\n', opts.sampleEvery);
fprintf('  %-40s %14s %14s %16s\n', 'candidate', 'with a return', 'in gaps', 'share < 0.1 mm');
score = inf(1, nC);
for c = 1:nC
    a = abs([dMeas{c}; dGap{c}]);
    a = a(isfinite(a));
    if isempty(a), continue; end
    score(c) = mean(a >= 1e-4) + mean(a);          % share not reproduced to 0.1 mm, then mean difference
    fprintf('  %-40s %12.5f m %12.5f m %16.3f\n', cands{c}, medf(abs(dMeas{c})), medf(abs(dGap{c})), mean(a < 1e-4));
end
[~, cBest] = min(score);
fprintf('  -> closest: %s\n', cands{cBest});

%% ---- 3. Detectors of filled stretches ---------------------------------------------
nMin = max(3, round(opts.gapMin / dx) - 1);       % filled cells in a gap of opts.gapMin
dets = struct('name', {}, 'kind', {}, 'k', {}, 'tol', {});
for k = [2 4]
    for tol = [1e-9 1e-7 1e-5]
        dets(end+1) = struct('name', sprintf('zero difference of order %d, tol %.0e m', k, tol), ...
            'kind', 'diff', 'k', k, 'tol', tol); %#ok<SAGROW>
    end
end
for thr = [1e-5 3e-5 1e-4 3e-4 1e-3]
    dets(end+1) = struct('name', sprintf('roughness below %.0e m', thr), 'kind', 'rough', 'k', 2, 'tol', thr); %#ok<SAGROW>
end
nD = numel(dets);
cntC = zeros(nD, 3);                               % near the crests: true positives, false positives, false negatives
cntW = zeros(nD, 3);                               % whole profiles
win = round(opts.crestWindow / dx);
pad = round(1.0 / dx);
for q = 1:size(crest, 1)
    i = crest(q, 1);
    r = crest(q, 2);
    [xr, ~] = rawScan(Tr, i);
    if numel(xr) < 3, continue; end
    rows = max(1, r - win - pad):min(nXs, r + win + pad);
    zc = double(zs(rows, i));
    [~, gapLen] = rawSupport(xs(rows), xr, dx);
    inW = abs(rows(:) - r) <= win & isfinite(zc) & xs(rows) >= xr(1) & xs(rows) <= xr(end);
    truth = gapLen >= opts.gapMin;
    for d = 1:nD
        F = detectFill(zc, dets(d), nMin);
        cntC(d, :) = cntC(d, :) + [sum(F & truth & inW), sum(F & ~truth & inW), sum(~F & truth & inW)];
    end
end
for i = scansA
    [xr, ~] = rawScan(Tr, i);
    if numel(xr) < 3, continue; end
    zc = double(zs(:, i));
    [~, gapLen] = rawSupport(xs, xr, dx);
    use = isfinite(zc) & xs >= xr(1) & xs <= xr(end);
    truth = gapLen >= opts.gapMin;
    for d = 1:nD
        F = detectFill(zc, dets(d), nMin);
        cntW(d, :) = cntW(d, :) + [sum(F & truth & use), sum(F & ~truth & use), sum(~F & truth & use)];
    end
end
fprintf('\nDetectors of filled stretches (truth: grid points in gaps of at least %.0f cm between returns)\n', 100 * opts.gapMin);
fprintf('  %-44s %26s   %26s\n', '', 'near the crests', 'whole profiles');
fprintf('  %-44s %8s %8s %8s   %8s %8s %8s\n', 'detector', 'precis.', 'recall', 'F1', 'precis.', 'recall', 'F1');
f1C = zeros(1, nD);
for d = 1:nD
    [pC, rC, f1C(d)] = prf(cntC(d, :));
    [pW, rW, fW] = prf(cntW(d, :));
    fprintf('  %-44s %8.3f %8.3f %8.3f   %8.3f %8.3f %8.3f\n', dets(d).name, pC, rC, f1C(d), pW, rW, fW);
end
[f1Best, dBest] = max(f1C);
useMask = f1Best >= opts.minF1;
fprintf('  -> best near the crests: %s (F1 %.3f); %s\n', dets(dBest).name, f1Best, ...
    ternary(useMask, 'used for the masks', sprintf('below %.2f, so no mask is written', opts.minF1)));

%% ---- 4. Where the returns fall around the crests, and the error of the fill ---------
edges = -opts.range:opts.bin:opts.range;
nB = numel(edges) - 1;
pat.xi   = (edges(1:end-1) + edges(2:end))' / 2;
pat.nObs = zeros(nB, 1);  pat.nMeas = zeros(nB, 1);  pat.nGap = zeros(nB, 1);
gEdges = [0.02 0.05 0.1 0.2 0.5 1 2 5];
gapNear = zeros(1, numel(gEdges) - 1);  gapNearLen = gapNear;  winLen = 0;
crestHit1 = 0;  crestHit2 = 0;
for q = 1:size(crest, 1)
    i = crest(q, 1);
    r = crest(q, 2);
    [xr, ~] = rawScan(Tr, i);
    if numel(xr) < 3, continue; end
    xi = xs - xs(r);
    in = abs(xi) < opts.range & xs >= xr(1) & xs <= xr(end);
    [d, gapLen, inCell] = rawSupport(xs(in), xr, dx);
    b = floor((xi(in) - edges(1)) / opts.bin) + 1;
    ok = b >= 1 & b <= nB;
    pat.nObs  = pat.nObs  + accumarray(b(ok), 1, [nB 1]);
    pat.nMeas = pat.nMeas + accumarray(b(ok), double(inCell(ok)), [nB 1]);
    pat.nGap  = pat.nGap  + accumarray(b(ok), double(gapLen(ok) >= opts.gapMin), [nB 1]);
    dc = distToRaw(xs(r), xr);
    crestHit1 = crestHit1 + (dc <= dx / 2);
    crestHit2 = crestHit2 + (dc <= 0.02);
    sel = xr >= xs(r) - opts.crestWindow & xr <= xs(r) + opts.crestWindow;
    g = diff(xr(sel));
    for e = 1:numel(gEdges) - 1
        m = g >= gEdges(e) & g < gEdges(e + 1);
        gapNear(e) = gapNear(e) + sum(m);
        gapNearLen(e) = gapNearLen(e) + sum(g(m));
    end
    winLen = winLen + 2 * opts.crestWindow;
end
nQ = size(crest, 1);
pat.Pmeas = pat.nMeas ./ max(pat.nObs, 1);
pat.Pgap  = pat.nGap ./ max(pat.nObs, 1);
pat.gapEdges = gEdges;
pat.gapCountNear = gapNear;
pat.gapLengthShareNear = gapNearLen / max(winLen, eps);
pat.crestMeasuredCell = crestHit1 / max(nQ, 1);
pat.crestMeasured2cm  = crestHit2 / max(nQ, 1);
pat.returnsPerScan = nRet(:);
fprintf('\nReturns around the crests (%d scans before onset):\n', nQ);
fprintf('  crest point with a return in its 1-cm cell: %.3f of the scans; within 2 cm: %.3f\n', ...
    pat.crestMeasuredCell, pat.crestMeasured2cm);
for sideName = {'front face (shoreward)', 'back face (seaward)'}
    if strcmp(sideName{1}(1), 'f'), s = pat.xi < 0 & pat.xi > -opts.crestWindow; else, s = pat.xi > 0 & pat.xi < opts.crestWindow; end
    fprintf('  %-24s within %.0f m: share of cells with a return %.3f, share in gaps of %.0f cm or more %.3f\n', ...
        sideName{1}, opts.crestWindow, sum(pat.nMeas(s)) / max(sum(pat.nObs(s)), 1), 100 * opts.gapMin, ...
        sum(pat.nGap(s)) / max(sum(pat.nObs(s)), 1));
end
fprintf('  gaps between returns within %.0f m of the crest: length (m), number per scan, share of the window\n', opts.crestWindow);
for e = 1:numel(gEdges) - 1
    fprintf('    %5.2f-%4.2f %10.2f %10.3f\n', gEdges(e), gEdges(e + 1), gapNear(e) / max(nQ, 1), pat.gapLengthShareNear(e));
end

% error of the fill: remove the returns over a stretch, grid again, compare at the cells that had returns
classes = {'crest', 'front face', 'back face'};
nL = numel(opts.removeL);
errRMS = nan(nL, 3);  errPeak = nan(nL, 1);  nDone = zeros(nL, 3);
for l = 1:nL
    L = opts.removeL(l);
    for cl = 1:3
        e2 = [];  pk = [];  tries = 0;
        while numel(e2) < opts.nTrials && tries < 20 * opts.nTrials && nQ > 0
            tries = tries + 1;
            q = randi(nQ);
            i = crest(q, 1);
            r = crest(q, 2);
            [xr, zr] = rawScan(Tr, i);
            if numel(xr) < 10, continue; end
            switch cl
                case 1, c0 = xs(r) + (rand - 0.5) * L / 2;
                case 2, c0 = xs(r) - (L / 2 + 0.05 + rand * max(opts.crestWindow - L / 2 - 0.05, 0));
                case 3, c0 = xs(r) + (L / 2 + 0.05 + rand * max(opts.crestWindow - L / 2 - 0.05, 0));
            end
            rows = find(abs(xs - xs(r)) <= opts.crestWindow + L + 1.0);
            cut = xr >= c0 - L / 2 & xr <= c0 + L / 2;
            if sum(cut) < 3 || sum(~cut) < 10, continue; end
            zFull = gridReturns(cands{cBest}, xs(rows), xr, zr);
            zCut  = gridReturns(cands{cBest}, xs(rows), xr(~cut), zr(~cut));
            [~, ~, inCell] = rawSupport(xs(rows), xr(cut), dx);
            m = inCell & abs(xs(rows) - c0) <= L / 2 & isfinite(zFull) & isfinite(zCut);
            if sum(m) < 3, continue; end
            e2(end+1) = sqrt(mean((zCut(m) - zFull(m)).^2)); %#ok<SAGROW>
            if cl == 1
                inG = abs(xs(rows) - c0) <= L / 2;
                pk(end+1) = max(zCut(inG)) - max(zFull(inG)); %#ok<SAGROW>
            end
        end
        errRMS(l, cl) = medf(e2);
        nDone(l, cl) = numel(e2);
        if cl == 1, errPeak(l) = medf(pk); end
    end
end
pat.fillErrL = opts.removeL(:);
pat.fillErrRMS = errRMS;
pat.fillErrClasses = classes;
pat.fillErrCrestPeak = errPeak;
pat.fillMethod = cands{cBest};
fprintf('\nError of the fill (%s): returns removed over a stretch of length L, gridded again;\n', cands{cBest});
fprintf('median RMS difference from the full gridding at the cells that had returns (m)\n');
fprintf('  %6s %10s %12s %12s %24s\n', 'L (m)', 'crest', 'front face', 'back face', 'crest height change (m)');
for l = 1:nL
    fprintf('  %6.2f %10.4f %12.4f %12.4f %24.4f   (%d/%d/%d trials)\n', opts.removeL(l), errRMS(l, 1), ...
        errRMS(l, 2), errRMS(l, 3), errPeak(l), nDone(l, 1), nDone(l, 2), nDone(l, 3));
end
pat.opts = opts;
save(fullfile(here, 'carini_lidar_pattern.mat'), 'pat', '-v7');
fprintf('Pattern saved to carini_lidar_pattern.mat\n');

%% ---- 5. Masks for the exported waves ------------------------------------------------
if useMask && exist(fileExp, 'file') == 2 && exist(fileWaves, 'file') == 2
    E = load(fileExp, 'waves');
    small = {'tcat', 'xcat', 'ztrprecat', 'ixcat'};
    S = load(fileWaves, small{:});
    fprintf('\nLoading the profiles (about 3 GB) ...\n');
    P = load(fileWaves, 'xprofilesCat', 'zprofilesCat');
    nE = numel(E.waves);
    mask = repmat(struct('id', NaN, 'gapFill', []), nE, 1);
    shareC = nan(nE, 1);
    for e = 1:nE
        k  = E.waves(e).id;
        t  = toEpoch(cellOf(S.tcat, k));
        nT = numel(t);
        [xg, eta] = rebuildWave(cellOf(P.xprofilesCat, k), cellOf(P.zprofilesCat, k), ...
            fieldVec(S, 'xcat', k), fieldVec(S, 'ztrprecat', k), nT);
        F = false(size(eta));
        for j = 1:nT, F(:, j) = detectFill(eta(:, j), dets(dBest), nMin); end
        ig = round((E.waves(e).x(:) - xg(1)) / medf(diff(xg))) + 1;
        ig = min(max(ig, 1), numel(xg));
        mask(e).id = k;
        mask(e).gapFill = F(ig, :);
        rc = fieldVec(S, 'ixcat', k);
        if E.waves(e).type >= 2 && ~isnan(E.waves(e).onset), sc = 1:E.waves(e).onset - 1; else, sc = 1:nT; end
        a = 0; b = 0;
        for j = sc
            if j > numel(rc) || rc(j) < 1 || rc(j) > numel(xg), continue; end
            rows = max(1, rc(j) - win):min(numel(xg), rc(j) + win);
            ok = isfinite(eta(rows, j));
            a = a + sum(F(rows, j) & ok);
            b = b + sum(ok);
        end
        if b > 0, shareC(e) = a / b; end
    end
    maskInfo.detector = dets(dBest).name;
    maskInfo.f1Crest = f1Best;
    maskInfo.gapMin = opts.gapMin;
    maskInfo.note = 'true where the gridded profile was probably filled across a gap between lidar returns';
    save(fullfile(here, 'carini_pinn_mask.mat'), 'mask', 'maskInfo', '-v7');
    ty = [E.waves.type]';
    fprintf('Masks written to carini_pinn_mask.mat for %d waves (%s)\n', nE, dets(dBest).name);
    fprintf('Share of the surface within %.0f m of the crest flagged as filled (scans before onset), median:\n', opts.crestWindow);
    fprintf('  spilling %.3f, plunging %.3f, non-breaking %.3f (example record: %.3f of the crest region in gaps)\n', ...
        medf(shareC(ty == 2)), medf(shareC(ty == 3)), medf(shareC(ty == 0)), ...
        sum(pat.nGap(abs(pat.xi) < opts.crestWindow)) / max(sum(pat.nObs(abs(pat.xi) < opts.crestWindow)), 1));
    clear P
elseif ~useMask
    fprintf('\nNo detector works well enough on the record: no mask written.\n');
else
    fprintf('\n%s or %s not found: no mask written.\n', fileExp, fileWaves);
end

%% ---- 6. Figure ---------------------------------------------------------------------
if opts.makeFigures
    try
        makeFigure(Tr, xs, zs, crest, pat, opts, here);
    catch err
        fprintf('Figure not made: %s\n', err.message);
    end
end

diary off;
fprintf('\nLog written to %s\n', logFile);


%% ======================== local functions =====================================

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end

function A = cellOf(V, k)
    if iscell(V), A = V{k}; else, A = V(k); end
end

function v = fieldVec(S, name, k)
    if isfield(S, name), v = double(cellOf(S.(name), k)); v = v(:); else, v = []; end
end

function v = at(a, j)
    if j <= numel(a), v = a(j); else, v = NaN; end
end

function m = medf(v)
% Median of the finite values (NaN if there are none).
    v = double(v(isfinite(v)));
    if isempty(v), m = NaN; else, m = median(v(:)); end
end

function t = toEpoch(t)
% Seconds since 1970-01-01 UTC from epoch seconds, milliseconds or datenum days.
    t = double(t(:));
    tm = medf(t);
    if tm > 1e11
        t = t / 1000;
    elseif tm > 7e5 && tm < 1e6
        t = (t - 719529) * 86400;
    end
end

function v = getScan(P, j, nT)
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

function [xg, eta] = rebuildWave(X, Z, xc, ztr, nT)
% Fixed FRF grid and free surface of one wave (profiles stored relative to crest and trough).
    xg  = getScan(X, 1, nT).' + at(xc, 1);
    eta = nan(numel(xg), nT);
    for j = 1:nT
        xj = getScan(X, j, nT).' + at(xc, j);
        zj = getScan(Z, j, nT).' + at(ztr, j);
        if max(abs(xj - xg)) < 1e-6
            eta(:, j) = zj;
        else
            ok = isfinite(xj) & isfinite(zj);
            if sum(ok) > 1, eta(:, j) = interp1(xj(ok), zj(ok), xg, 'linear', NaN); end
        end
    end
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

function d = distToRaw(xq, xr)
% Distance from each point xq to the nearest return xr.
    if numel(xr) < 2, d = inf(size(xq)); return; end
    d = abs(xq - interp1(xr, xr, xq, 'nearest', 'extrap'));
end

function [d, gapLen, inCell] = rawSupport(xq, xr, dx)
% Distance to the nearest return, length of the interval between the returns on either side
% (0 at a return, NaN outside), and whether a return lies in the 1-cm cell of each point.
    xq = xq(:);
    if numel(xr) < 2
        d = inf(size(xq)); gapLen = nan(size(xq)); inCell = false(size(xq)); return
    end
    d = distToRaw(xq, xr);
    gapLen = interp1(xr, xr, xq, 'next') - interp1(xr, xr, xq, 'previous');
    inCell = d <= dx / 2;
end

function zg = gridReturns(name, xq, xr, zr)
% Candidate gridding of the returns (xr, zr) onto the points xq.
    xq = xq(:);
    dx = medf(diff(xq));
    zg = nan(size(xq));
    if numel(xr) < 2, return; end
    span = xq >= xr(1) & xq <= xr(end);
    switch name
        case 'returns, straight lines'
            zg = interp1(xr, zr, xq, 'linear', NaN);
        case 'returns, pchip'
            zg(span) = interp1(xr, zr, xq(span), 'pchip');
        otherwise
            if strncmp(name, 'left-aligned', 12)
                b = floor((xr - xq(1)) / dx + 1e-9) + 1;
            else
                b = round((xr - xq(1)) / dx) + 1;
            end
            ok = b >= 1 & b <= numel(xq);
            if strncmp(name, 'cell medians', 12)
                zb = accumarray(b(ok), zr(ok), [numel(xq) 1], @median, NaN);
            else
                zb = accumarray(b(ok), zr(ok), [numel(xq) 1], @mean, NaN);
            end
            if strncmp(name, 'smoothed', 8)
                f = isfinite(zb);
                z0 = zb;
                z0(~f) = 0;
                s = conv(z0, [1; 1; 1], 'same') ./ max(conv(double(f), [1; 1; 1], 'same'), 1);
                zb(f) = s(f);
            end
            if ~isempty(strfind(name, 'straight-line fill')), method = 'linear'; %#ok<STREMP>
            elseif ~isempty(strfind(name, 'pchip fill')), method = 'pchip'; %#ok<STREMP>
            else, method = 'cubic';
            end
            zg = fillCells(zb, method);
    end
end

function z = fillCells(z, method)
% Fills the empty cells between the first and the last cell with a value.
    f = find(isfinite(z));
    if numel(f) < 2, return; end
    lo = f(1);
    hi = f(end);
    gapI = find(~isfinite(z(lo:hi))) + lo - 1;
    if isempty(gapI), return; end
    if strcmp(method, 'cubic')
        z = fillDel4(z, gapI);
    else
        z(gapI) = interp1(f, z(f), gapI, method);
    end
end

function z = fillDel4(z, gapI)
% Fills the cells gapI so that the fourth difference vanishes at each of them: a cubic across
% every gap (the del^4 option of inpaint_nans in one dimension). Near the ends of the array, the
% second difference is used instead.
    n = numel(z);
    m = numel(gapI);
    map = zeros(n, 1);
    map(gapI) = 1:m;
    I = zeros(5 * m, 1);  J = I;  V = I;  rhs = zeros(m, 1);  p = 0;
    for q = 1:m
        i = gapI(q);
        if i - 2 >= 1 && i + 2 <= n
            st = [1 -4 6 -4 1];  off = -2:2;
        else
            st = [1 -2 1];  off = -1:1;
        end
        for s = 1:numel(off)
            jj = i + off(s);
            if jj < 1 || jj > n, continue; end
            if map(jj) > 0
                p = p + 1;  I(p) = q;  J(p) = map(jj);  V(p) = st(s);
            elseif isfinite(z(jj))
                rhs(q) = rhs(q) - st(s) * z(jj);
            end
        end
    end
    A = sparse(I(1:p), J(1:p), V(1:p), m, m);
    z(gapI) = A \ rhs;
end

function F = detectFill(z, det, nMin)
% Points of a gridded profile z that look filled: centres of vanishing k-th differences, or
% locally smooth stretches, in runs of at least nMin points.
    z = z(:);
    n = numel(z);
    c = false(n, 1);
    switch det.kind
        case 'diff'
            k = det.k;
            if n > k
                d = abs(diff(z, k));
                c((1:numel(d)) + k / 2) = d < det.tol;
            end
        case 'rough'
            d2 = nan(n, 1);
            if n > 2, d2(2:n-1) = abs(diff(z, 2)); end
            M = [nan(2, 1); d2(1:end-2)];
            M = [M, [nan; d2(1:end-1)], d2, [d2(2:end); nan], [d2(3:end); nan(2, 1)]];
            Ssort = sort(M, 2);
            nf = sum(isfinite(M), 2);
            idx = max(1, ceil(nf / 2));
            r = Ssort(sub2ind(size(Ssort), (1:n)', idx));
            r(nf == 0) = NaN;
            c = r < det.tol;
    end
    c = c & isfinite(z);
    F = false(n, 1);
    e = diff([false; c; false]);
    b = find(e == 1);
    f = find(e == -1) - 1;
    for q = 1:numel(b)
        if f(q) - b(q) + 1 >= nMin, F(b(q):f(q)) = true; end
    end
end

function [p, r, f1] = prf(cnt)
% Precision, recall and F1 from [true positives, false positives, false negatives].
    p = cnt(1) / max(cnt(1) + cnt(2), 1);
    r = cnt(1) / max(cnt(1) + cnt(3), 1);
    f1 = 2 * p * r / max(p + r, eps);
end

function makeFigure(Tr, xs, zs, crest, pat, opts, here)
% (a) one crest with its returns and the gridded profile, gaps shaded; (b) share of the surface
% with a return and in gaps around the crest; (c) error of the fill against gap length.
    fig = figure('Color', 'w', 'Position', [100 100 1650 520]);
    set(fig, 'DefaultTextInterpreter', 'latex', 'DefaultAxesTickLabelInterpreter', 'latex', ...
        'DefaultLegendInterpreter', 'latex', 'DefaultAxesFontSize', 18, 'DefaultLegendFontSize', 15);
    tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    % (a) the crest scan with the largest share of the crest region in gaps, among the median half
    dx = medf(diff(xs));
    share = nan(size(crest, 1), 1);
    for q = 1:size(crest, 1)
        [xr, ~] = rawScan(Tr, crest(q, 1));
        rows = abs(xs - xs(crest(q, 2))) <= opts.crestWindow;
        [~, gl] = rawSupport(xs(rows), xr, dx);
        share(q) = mean(gl >= opts.gapMin);
    end
    [~, q] = min(abs(share - medf(share)));
    i = crest(q, 1);
    r = crest(q, 2);
    [xr, zr] = rawScan(Tr, i);
    ax = nexttile(tl);
    hold(ax, 'on');
    rows = find(abs(xs - xs(r)) <= 4);
    [~, gl] = rawSupport(xs(rows), xr, dx);
    zl = [min(double(zs(rows, i))) - 0.1, max(double(zs(rows, i))) + 0.1];
    g = gl >= opts.gapMin;
    e = diff([false; g(:); false]);
    b = find(e == 1);
    f = find(e == -1) - 1;
    for k = 1:numel(b)
        x0 = xs(rows(b(k))) - xs(r);
        x1 = xs(rows(f(k))) - xs(r);
        patch(ax, [x0 x1 x1 x0], [zl(1) zl(1) zl(2) zl(2)], [0.88 0.88 0.88], 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
    patch(ax, NaN, NaN, [0.88 0.88 0.88], 'EdgeColor', 'none', 'DisplayName', sprintf('gaps $\\geq %.0f$ cm', 100 * opts.gapMin));
    sel = xr >= xs(rows(1)) & xr <= xs(rows(end));
    plot(ax, xr(sel) - xs(r), zr(sel), '.', 'Color', [0.00 0.45 0.74], 'MarkerSize', 6, 'DisplayName', 'lidar returns');
    plot(ax, xs(rows) - xs(r), double(zs(rows, i)), '-', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.8, 'DisplayName', 'gridded surface');
    xlim(ax, [-4 4]);  ylim(ax, zl);  xticks(ax, -4:2:4);
    xlabel(ax, '$x - x_c$ (m)');  ylabel(ax, '$z$ (m)');
    legend(ax, 'Location', 'northoutside', 'Orientation', 'horizontal', 'NumColumns', 2, 'Box', 'off');
    styleAxes(ax);  panelLabel(ax, '(a)');

    % (b) share of the surface with a return in its cell, and in gaps, around the crest
    ax = nexttile(tl);
    hold(ax, 'on');
    plot(ax, pat.xi, pat.Pmeas, '-', 'Color', [0.00 0.45 0.74], 'LineWidth', 2, 'DisplayName', 'return in the 1-cm cell');
    plot(ax, pat.xi, pat.Pgap, '-', 'Color', [0.85 0.33 0.10], 'LineWidth', 2, ...
        'DisplayName', sprintf('in a gap $\\geq %.0f$ cm', 100 * opts.gapMin));
    xline(ax, 0, ':k', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    xlim(ax, [-opts.range opts.range]);  ylim(ax, [0 1]);  xticks(ax, -8:4:8);  yticks(ax, 0:0.2:1);
    xlabel(ax, '$x - x_c$ (m)');  ylabel(ax, 'Share of scans');
    legend(ax, 'Location', 'northoutside', 'Orientation', 'horizontal', 'NumColumns', 1, 'Box', 'off');
    styleAxes(ax);  panelLabel(ax, '(b)');

    % (c) error of the fill against the length of the removed returns
    ax = nexttile(tl);
    hold(ax, 'on');
    cols = [0.85 0.33 0.10; 0.00 0.45 0.74; 0.47 0.67 0.19];
    mk = {'o', 's', '^'};
    for c = 1:3
        plot(ax, pat.fillErrL, 100 * pat.fillErrRMS(:, c), ['-' mk{c}], 'Color', cols(c, :), 'LineWidth', 1.8, ...
            'MarkerFaceColor', cols(c, :), 'DisplayName', pat.fillErrClasses{c});
    end
    set(ax, 'XScale', 'log');
    xlim(ax, [0.08 2]);  xticks(ax, opts.removeL);
    xlabel(ax, 'Length of the removed returns (m)');  ylabel(ax, 'RMS error of the fill (cm)');
    legend(ax, 'Location', 'northoutside', 'Orientation', 'horizontal', 'NumColumns', 3, 'Box', 'off');
    styleAxes(ax);  panelLabel(ax, '(c)');

    base = fullfile(here, 'carini_gap_study');
    try
        exportgraphics(fig, [base '.png'], 'Resolution', 200);
        exportgraphics(fig, [base '.pdf'], 'ContentType', 'vector');
    catch
        print(fig, [base '.png'], '-dpng', '-r200');
    end
end

function styleAxes(ax)
    box(ax, 'on');
    set(ax, 'LineWidth', 1.8, 'FontSize', 18, 'TickLabelInterpreter', 'latex', 'Layer', 'top');
end

function panelLabel(ax, s)
    text(ax, 0.97, 0.95, ['\textbf{' s '}'], 'Units', 'normalized', 'Interpreter', 'latex', ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontSize', 20);
end
