module tfidf

using SparseArrays

export TextMatrix, Tfidf, 
       fit!, transform, fit_transform, get_col_names, 
       idf


###### include ########

include("types.jl")
include("construction_helpers.jl")
include("apply.jl")
include("utils.jl")


end # module



# Examples
documents = ["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]
