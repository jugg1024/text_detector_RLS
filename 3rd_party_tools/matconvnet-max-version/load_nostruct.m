function s = load_nostruct(filename)

    s = load(filename);
    if ~isstruct(s)
        return
    end
    fnames = fieldnames(s);
    if length(fnames) == 1
        s = s.(fnames{1});
    else
       % warning('load_nostruct: Loaded file is a struct with more than two fields. Returning struct');
    end
