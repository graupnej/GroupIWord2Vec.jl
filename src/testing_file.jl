using Plots
using MultivariateStats

gr()
function points_to_scatter(point...)
    xs = [i[1] for i in point]
    ys = [i[2] for i in point]
    return xs, ys
end

function  generate_distance_arrows(points::Tuple{Vector{Int64}, Vector{Int64}}...)
    x_mag = [(i[2]-i[1])[1] for i in points]
    y_mag = [(i[2]-i[1])[2] for i in points]
    
    return x_mag, y_mag
end


point1 = [2, 3]
point2 = [4, 5]
point3 = [3, 1]
point4 = [10, 3]

xs, ys =        points_to_scatter(point1, point2, point3, point4)
x_mag, y_mag =  generate_distance_arrows((point1, point2), (point3, point4))

scatter(xs, ys, series_annotations = text.(1:length(xs), :bottom))
quiver!(xs, ys, quiver=([Bool(i%2) ? x_mag[Int(i/2 + 0.5)] : 0 for i in 1:(length(x_mag)*2)], [Bool(i%2) ? y_mag[Int(i/2 + 0.5)] : 0 for i in 1:(length(y_mag)*2)]))




