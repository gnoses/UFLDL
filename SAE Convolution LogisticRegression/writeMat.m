function writeMat(var)
% var = ones(12,12);
% filename = 'ZCAWhite.bin';
% a = ones(10);

filename = sprintf('%s.bin', inputname(1));
fid = fopen(filename,'w');
dim = size(var);
dimCount = size(dim,2);
fwrite(fid, dimCount, 'int');
fwrite(fid, dim, 'int');
fwrite(fid, var, 'double');
fclose(fid);
