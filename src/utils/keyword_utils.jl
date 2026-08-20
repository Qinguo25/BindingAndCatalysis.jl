struct _UnsetKeyword end
const _UNSET_KEYWORD = _UnsetKeyword()

@noinline function _renamed_keyword_error(old::Symbol, new::Symbol)
    throw(ArgumentError("keyword `$(old)` is no longer supported; use `$(new)` instead."))
end

@inline function _reject_renamed_keywords(kwargs)
    haskey(kwargs, :recalculate) && _renamed_keyword_error(:recalculate, :recompute)
    haskey(kwargs, :abs_tol) && _renamed_keyword_error(:abs_tol, :abstol)
    haskey(kwargs, :rel_tol) && _renamed_keyword_error(:rel_tol, :reltol)
    return nothing
end

@inline function _reject_renamed_tolerance_keywords(abs_tol, rel_tol)
    abs_tol === _UNSET_KEYWORD || _renamed_keyword_error(:abs_tol, :abstol)
    rel_tol === _UNSET_KEYWORD || _renamed_keyword_error(:rel_tol, :reltol)
    return nothing
end

@inline function _reject_stability_keywords(kwargs)
    _reject_renamed_keywords(kwargs)
    haskey(kwargs, :return_code) || return nothing
    throw(
        ArgumentError(
            "keyword `return_code` is no longer supported; call `stability_code(...)` instead.",
        ),
    )
end
