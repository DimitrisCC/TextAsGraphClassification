import itertools


def terms_to_graph(terms, w):  # terms=list w=window size

    from_to = {}

    # create initial graph (first w terms)
    terms_temp = terms[0:w]
    indexes = list(itertools.combinations(range(w), r=2))

    new_edges = list()

    for i in range(len(indexes)):
        new_edges.append(" ".join(list(terms_temp[i] for i in indexes[i])))

    for i in range(0, len(new_edges)):
        from_to[new_edges[i].split()[0], new_edges[i].split()[1]] = 1

    # then iterate over the remaining terms
    for i in range(w, len(terms)):
        # term to consider
        considered_term = terms[i]
        # all terms within sliding window
        terms_temp = terms[(i-w+1):(i+1)]

        # edges to try
        candidate_edges = list()
        for p in range(w-1):
            candidate_edges.append((terms_temp[p], considered_term))

        for try_edge in candidate_edges:

            # if not self-edge
            if try_edge[1] != try_edge[0]:

                boolean1 = (try_edge[0], try_edge[1]) in from_to
                boolean2 = (try_edge[1], try_edge[0]) in from_to

                # if edge has already been seen, update its weight
                if boolean1:
                    from_to[try_edge[0], try_edge[1]] += 1

                elif boolean2:
                    from_to[try_edge[1], try_edge[0]] += 1

                # if edge has never been seen, create it and assign it a unit weight
                else:
                    from_to[try_edge] = 1
