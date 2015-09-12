function [img] = normalizeHorIntensity(img_raw)
% Remove low frequency variations across the image
sigma = 16;

% Remove DC component
horAvg = mean(img_raw,1);
avg = mean(horAvg);
horAvg = horAvg - avg;

% Pad signal
lenX = length(horAvg);
horAvgPadded = [ones(1,2*sigma)*mean(horAvg(1:4)) horAvg ones(1,2*sigma)*mean(horAvg(lenX-3:lenX))];

% Smooth average
freqHorAvg = fft(horAvgPadded);
halfLen = length(horAvgPadded) / 2;

freqKernel = exp(-[0:halfLen-1 -halfLen:-1].^2 * sigma / length(horAvgPadded));
lowPassCompPadded = ifft(freqHorAvg .* freqKernel);
lowPassComp = lowPassCompPadded(2*sigma+1:2*sigma+lenX);
img = img_raw - ones(size(img_raw,1), 1) * lowPassComp;
img = single(img * 130.0 / mean(img(:)));