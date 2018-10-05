function idf(df, n_docs)
    log((1+n_docs)/(1+df)) + 1
end

function l2(vec)
    quot = sqrt(sum(vec.^2))
    vec ./ quot
end

function l1(vec)
    quot = sum(map(abs, vec))
    vec ./ quot
end

function tokenize_simple(s::String)
    split(s)
end
