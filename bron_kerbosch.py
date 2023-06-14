def max_cliques(neighbors):  # dict[sets]
    def bron_kerbosch_1(clique, candidates, excluded):
        if len(clique) >= 2 and not candidates and not excluded:
            cliques.append(clique)
        for vertex in candidates:
            neighs = neighbors[vertex]
            bron_kerbosch_1(clique | {vertex}, candidates & neighs, excluded & neighs)
            candidates = candidates - {vertex}
            excluded = excluded | {vertex}
    cliques = []
    bron_kerbosch_1(set(), set(neighbors), set())
    return cliques
