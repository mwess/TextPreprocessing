abstract type TextMatrix end

mutable struct Tfidf{R<:Real, 
                     S<:AbstractFloat, 
                     T<:AbstractString, 
                    } <: TextMatrix
    min_df::R
    max_df::S
    ngrams::Tuple{Integer, Integer}
    lower::Bool
    stoplist::Union{AbstractVector{T}, Nothing}
    tokenizer
    norm
    idf::Union{Dict{AbstractString, Real} ,Nothing}
    tm::Union{Dict{AbstractString, Integer}, Nothing}
end

# Outer constructor
function Tfidf(min_df::R, 
               max_df::S, 
               ngrams::Tuple{T,T}; 
               lower::Bool=true,
               stoplist::Union{Vector{String}, Nothing}=nothing,
               norm::Symbol=:l2,
               tokenizer=tokenize_simple
              ) where {R<:Real, S<:Real, T<:Integer}
    if !(typeof(max_df) <: AbstractFloat)
        max_df = AbstractFloat(max_df)
    end
    if !(0 < max_df <= 1)
        error("max_df needs to be in (0,1].")
    end
    if !(ngrams[1] <= ngrams[2]) || ngrams[1] < 1
        error("Tuple parameter is wrong!")
    end
    if typeof(stoplist) == Nothing
        stoplist = Vector{AbstractString}()
    end
    if norm == :l2
        norm_fun = l2
    elseif norm == :l1
        norm_fun = l1
    end
    Tfidf{R, typeof(max_df), eltype(stoplist)}(min_df, max_df, ngrams, lower, stoplist, tokenizer, norm_fun, nothing, nothing)
end
