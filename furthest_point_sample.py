import math
import random
import numpy as np


def fps(vertices, cluster_cnt, seed_cnt):
    num_vertex = vertices.shape[0]

    distance = np.ones(num_vertex) * 1e10

    # initial random centroids
    centroids = random.sample([i for i in xrange(num_vertex)], seed_cnt)
    for center in centroids:
        furthest = update_distance(distance, vertices, center)

    # iteratively find furthest points
    while len(centroids) < cluster_cnt:
        centroids.append(furthest)
        #print "Add {0}th sample point".format(len(centroids))
        furthest = update_distance(distance, vertices, furthest)

    return np.array(centroids)


def update_distance(distance, vertices, center_idx):
    num_vertex = vertices.shape[0]

    max_dis = -1
    furthest = -1
    for p in xrange(num_vertex):
        dis = dpp(vertices[p], vertices[center_idx])
        if dis < distance[p]:
            distance[p] = dis

        if distance[p] > max_dis:
            max_dis = distance[p]
            furthest = p

    return furthest


def assign_faces(faces, vertices, centroid_indices):
    num_faces = faces.shape[0]
    distance = np.ones(num_faces) * 1e10
    belongs = np.zeros(num_faces)

    num_centroid = centroid_indices.shape[0]
    centroids = vertices[centroid_indices]

    # calculate face centroids
    for i in xrange(num_faces):
        # get face centroid
        face = vertices[faces[i]]
        face_centroid = np.zeros(3, dtype=np.float32)
        for j in xrange(3):
            for k in xrange(3):
                face_centroid[k] += face[j][k]
        face_centroid = face_centroid / 3

        # calc closest centroid
        for j in xrange(num_centroid):
            dis = dpp(face_centroid, centroids[j])
            if dis < distance[i]:
                distance[i] = dis
                belongs[i] = j

    return belongs


# distance between point & point
def dpp(p1, p2):
    dis = 0
    for i in xrange(3):
        dis += (p1[i] - p2[i])**2

    return math.sqrt(dis)

    # def surface_fps(vertices, faces, cluster_count, seed_count):
    #     vertex_count = len(vertices)
    #     face_count = len(faces)

    #     # build graph
    #     conn = [{} for i in xrange(vertex_count)]
    #     for j in xrange(face_count):
    #         face = faces[j][0]
    #         count = len(face)
    #         for i in xrange(count):
    #             if not face[i + 1 - count] - 1 in conn[face[i] - 1]:
    #                 conn[face[i] - 1][face[i + 1 - count] - 1] = d(vertices[face[i] - 1], vertices[face[i + 1 - count] - 1])
    #             if not face[i - 1] - 1 in conn[face[i] - 1]:
    #                 conn[face[i] - 1][face[i - 1] - 1] = d(vertices[face[i] - 1], vertices[face[i - 1] - 1])

    #     distance = [1e10 for i in xrange(vertex_count)]
    #     belong = [-1 for i in xrange(vertex_count)]

    #     # initial random centroids
    #     centroids = random.sample([i for i in xrange(vertex_count)], seed_count)
    #     for center in centroids:
    #         furthest = update_distance(distance, belong, conn, center)

    #     # iteratively find furthest points
    #     while len(centroids) < cluster_count:
    #         centroids.append(furthest)
    #         print "Add {0}th sample point".format(len(centroids))
    #         furthest = update_distance(distance, belong, conn, furthest)

    #     return centroids, belong

    # def d(p1, p2):
    #     return 1

    # def d(p1, p2):
    #     dis = 0
    #     for k in xrange(3):
    #         dis += (p1[k] - p2[k])**2
    #     return math.sqrt(dis)

    # def update_distance(distance, belong, conn, center):
    #     vertex_count = len(distance)
    #     inqueue = [False for i in xrange(vertex_count)]
    #     distance[center] = 0
    #     inqueue[center] = True
    #     queue, l, r = [center], -1, 0
    #     while l < r:
    #         l += 1
    #         u = queue[l]
    #         inqueue[u] = False
    #         for v in conn[u]:
    #             if distance[u] + conn[u][v] < distance[v]:
    #                 distance[v] = distance[u] + conn[u][v]
    #                 belong[v] = center
    #                 if not inqueue[v]:
    #                     queue.append(v)
    #                     inqueue[v] = True
    #                     r += 1

    #     furthest = -1
    #     for i in xrange(vertex_count):
    #         if furthest == -1 or distance[i] > distance[furthest]:
    #             furthest = i
    #     return furthest

    # def update_distance(distance, belong, conn, center):
    #     vertex_count = len(distance)
    #     inqueue = [False for i in xrange(vertex_count)]
    #     cdist = [1e10 for i in xrange(vertex_count)]
    #     cdist[center] = 0
    #     inqueue[center] = True
    #     queue, l, r = [center], -1, 0
    #     while l < r:
    #         l += 1
    #         u = queue[l]
    #         inqueue[u] = False
    #         for v in conn[u]:
    #             if cdist[u] + conn[u][v] < cdist[v]:
    #                 cdist[v] = cdist[u] + conn[u][v]
    #                 if not inqueue[v]:
    #                     queue.append(v)
    #                     inqueue[v] = True
    #                     r += 1

    #     furthest = -1
    #     for i in xrange(vertex_count):
    #         if cdist[i] < distance[i]:
    #             distance[i] = cdist[i]
    #             belong[i] = center
    #         if furthest == -1 or distance[i] > distance[furthest]:
    #             furthest = i
    #     return furthest
