
fdir = '/unrshare/LESCROARTSHARE/data_PSY763/SnowLabData/';
d = dir([fdir '*vtc']);
files = fullfile(fdir, {d.name})';
for i = 1:length(files);
    fprintf('Processing %s\n', files{i});
    vtc = xff(files{i});
    data = vtc.VTCData;
    new_file = strrep(files{i}, '.vtc', '.mat');
    save(new_file, 'data', '-v7.3');
end
