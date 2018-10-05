function fit2!(tf::Tfidf, documents::AbstractVector{T}) where {T <: AbstractString}
    if tf.lower
        documents = map(lowercase, documents)
    end
    documents_tok = map(tf.tokenizer, documents)
    ignore_list = compute_ignore_list(documents_tok, tf.min_df, tf.max_df, tf.stoplist)
    df = Dict{eltype(documents_tok[1]), Real}()
    for doc in documents_tok
        tmp_df = Dict{eltype(documents_tok[1]), Real}()
        tokens = filter(token -> !(token in ignore_list), doc)
        for i=tf.ngrams[1]:tf.ngrams[2]
            for j=1:length(tokens)-i
                l_idx = join(tokens[j:j+i-1], " ")
                if !(l_idx in keys(tmp_df))
                    tmp_df[l_idx] = 1
                end
            end
        end
        for key in keys(tmp_df)
            df[key] = tmp_df[key] + get(df, key, 0)
        end
    end
    n_docs = length(documents)
    for token in keys(df)
        df[token] = idf(df[token], n_docs)
    end
    ind_map = Dict{eltype(documents_tok[1]), Integer}(x[1] => x[2] for x in zip(keys(df),1:length(df)))
    tf.idf = df
    tf.tm = ind_map
    tf
end

function fit!(tf::Tfidf, documents::AbstractVector{T}) where {T <: AbstractString}
    if tf.lower
        documents = map(lowercase, documents)
    end
    documents_tok = map(tf.tokenizer, documents)
    ignore_list = compute_ignore_list(documents_tok, tf.min_df, tf.max_df, tf.stoplist)
    df = Dict{eltype(documents_tok[1]), Real}()
    for (ind, doc) in enumerate(documents_tok)
        tmp_df = Dict{eltype(documents_tok[1]), Real}()
        tokens = filter(token -> !(token in ignore_list), doc)
        for i=tf.ngrams[1]:tf.ngrams[2]
            for j=1:length(tokens)-i
                l_idx = join(tokens[j:j+i-1], " ")
                if !haskey(tmp_df, l_idx)
                    tmp_df[l_idx] = 1
                end
            end
        end
        for key in keys(tmp_df)
            df[key] = tmp_df[key] + get(df, key, 0)
        end
    end
    n_docs = length(documents)
    for token in keys(df)
        df[token] = idf(df[token], n_docs)
    end
    ind_map = Dict{eltype(documents_tok[1]), Integer}(x[1] => x[2] for x in zip(keys(df),1:length(df)))
    tf.idf = df
    tf.tm = ind_map
    tf
end

function transform(tf::Tfidf, documents::AbstractVector{T}) where {T <: AbstractString}
    if tf.idf == nothing
        error("fit! needs to be executed before transform.")
    end
    if tf.lower
        documents = map(lowercase, documents)
    end
    documents_tok = map(tf.tokenizer, documents)
    n_docs = length(documents_tok)
    term_matrix = spzeros(n_docs, length(tf.tm))
    for (doc_ind, doc) in enumerate(documents_tok)
        for i=tf.ngrams[1]:tf.ngrams[2]
            for j=1:length(doc)-i
                l_idx = join(doc[j:j+i-1], " ")
                if l_idx in keys(tf.idf)
                    term_matrix[doc_ind, tf.tm[l_idx]] += 1
                end
            end
        end
    end
    for i=1:length(n_docs)
        for key in keys(tf.idf)
            term_matrix[i, tf.tm[key]] *= tf.idf[key]
        end
    end
    term_matrix
end

function fit_transform(tf::Tfidf, documents::AbstractVector{T}) where {T <: AbstractString}
    fit!(tf, documents)
    transform(tf, documents)
end

function get_col_names(frequencies)
    map(keys, frequencies) |> x -> map(y -> map( z -> join(z, " "), collect(y)), x) |> x -> Iterators.flatten(x) |> x -> unique(x)
    map(keys, frequencies) |> x-> Iterators.flatten(x) |> x -> unique(x)
end
