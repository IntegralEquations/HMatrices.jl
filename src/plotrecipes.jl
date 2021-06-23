@recipe function f(rec::HyperRectangle{N}) where {N}
    seriestype := :path
    linecolor --> :black
    linestyle --> :solid
    label --> ""
    if N == 2
        pt1 = rec.low_corner
        pt2 = rec.high_corner
        x1, x2 = pt1[1], pt2[1]
        y1, y2 = pt1[2], pt2[2]
        [x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1]
    elseif N == 3
        seriestype := :path
        pt1 = rec.low_corner
        pt2 = rec.high_corner
        x1, x2 = pt1[1], pt2[1]
        y1, y2 = pt1[2], pt2[2]
        z1, z2 = pt1[3], pt2[3]
        # upper and lower faces
        @series [x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1], [z1,z1,z1,z1,z1]
        @series [x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1], [z2,z2,z2,z2,z2]
        # lines connecting faces
        @series [x1,x1], [y1,y1], [z1,z2]
        @series [x2,x2], [y1,y1], [z1,z2]
        @series [x2,x2], [y2,y2], [z1,z2]
        @series [x1,x1], [y2,y2], [z1,z2]
    end
end

"""
    plot(tree::ClusterTree,args...)

Plot the point could and the bounding boxes at the leaves of the tree
"""
plot(tree::ClusterTree,args...) = ()

@recipe function f(tree::ClusterTree;filter=(x)->isleaf(x))
    legend := false
    grid   --> false
    aspect_ratio --> :equal
    # plot points
    N = dimension(tree)
    if N == 2
        @series begin
            seriestype := :scatter
            markersize := 2
            xx = [pt[1] for pt in tree.points]
            yy = [pt[2] for pt in tree.points]
            xx,yy
        end
    elseif N == 3
        @series begin
            seriestype := :scatter
            markersize := 2
            xx = [pt[1] for pt in tree.points]
            yy = [pt[2] for pt in tree.points]
            zz = [pt[3] for pt in tree.points]
            xx,yy,zz
        end
    end
    # plot bounding boxes
    blocks = getblocks(filter,tree)
    for block in blocks
        @series begin
            linestyle --> :solid
            seriescolor  --> :black
            block.bounding_box
        end
    end
end
