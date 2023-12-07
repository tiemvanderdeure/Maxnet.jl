function hinge(x, nknots = 50)
    mi, ma = extrema(x)
    k = range(mi, ma; length = nknots)
    lh = hingeval.(x, k[1:end-1]', [ma])
    rh = hingeval.(x, [mi], k[2:end]')
    [lh rh]
end
hingeval(x, mi, ma) = clamp((x - mi) / (ma - mi), 0, 1)

function thresholds(x, nknots = 50)
    k = range(extrema(x)...; length = nknots + 2)[2:nknots + 1]
    Bool.(x .>= k')
end

