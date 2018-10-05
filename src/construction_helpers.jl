function compute_ignore_list(corpus_tok, min_df, max_df, stoplist)
    tokens = unique(vcat(corpus_tok...))
    doc_freqs = Dict{eltype(corpus_tok[1]), Integer}(token => 0 for token in tokens)
    for doc in corpus_tok
        for key in unique(doc)
            doc_freqs[key] += 1
        end
    end
    if min_df >= 1
        min_df = min_df/length(corpus_tok)
    end
    ignore_list = Vector{eltype(corpus_tok[1])}()
    for key in keys(doc_freqs)
        token_freq = doc_freqs[key]/length(corpus_tok)
        if !(min_df < token_freq < max_df)
            push!(ignore_list, key)
        end
    end
    Set(vcat(ignore_list, stoplist))
end

function convert_to_sparse_matrix(frequencies)
    col_names = collect(get_col_names(frequencies))
    col_idxs = Dict(zip(col_names, 1:length(col_names)))
    #col_idxs_rev = Dict(zip(1:length(col_names), col_names))
    n_docs = length(frequencies)
    sparse_mat = spzeros(length(col_idxs), n_docs)
    for i=1:length(frequencies)
        for key in keys(frequencies[i])
            sparse_mat[col_idxs[key], i] = frequencies[i][key]
        end
    end
    sparse_mat, col_idxs
end
