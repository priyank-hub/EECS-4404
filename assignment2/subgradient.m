function v = subgradient(w, x, t)
if 1 - t * dot(w, x) <= 0
    v = 0;
elseif 1 - t * dot(w, x) > 0
    v = -t*x;
end