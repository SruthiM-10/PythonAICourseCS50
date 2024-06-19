import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():
    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    joint_prob = 1
    flag = False
    # looping through the people who don't have parents
    for person in people.keys():
        if person in one_gene and person in two_genes:
            flag = True
        if people[person]["mother"] is None or people[person]["father"] is None:
            if person in one_gene:
                joint_prob *= PROBS["gene"][1]
                if person in have_trait:
                    joint_prob *= PROBS["trait"][1][True]
                else:
                    joint_prob *= PROBS["trait"][1][False]
            elif person in two_genes:
                joint_prob *= PROBS["gene"][2]
                if person in have_trait:
                    joint_prob *= PROBS["trait"][2][True]
                else:
                    joint_prob *= PROBS["trait"][2][False]
            else:
                joint_prob *= PROBS["gene"][0]
                if person in have_trait:
                    joint_prob *= PROBS["trait"][0][True]
                else:
                    joint_prob *= PROBS["trait"][0][False]
    # looping through the children
    for person in people.keys():
        if people[person]["mother"] is not None:
            mother = people[person]["mother"]
            father = people[person]["father"]
            if person in one_gene:
                # two cases where they get the gene from mother but not father, and vice versa
                if person in have_trait:
                    joint_prob *= PROBS["trait"][1][True]
                else:
                    joint_prob *= PROBS["trait"][1][False]
                if mother in one_gene:
                    # In 1st case -> Mother: (right gene copy gets passed down and doesn't get mutated or wrong gene copy gets passed down and is mutated)
                    # In 2nd case -> Mother: (right gene copy gets passed down and gets mutated or wrong gene copy gets passed down and is not mutated)
                    # 0.5 * 0.99 + 0.5 * 0.1 = 0.5 in both cases
                    if father in one_gene:
                        joint_prob *= (0.5 * 0.5 + 0.5 * 0.5)
                        # Opposite case for father but same probability
                    elif father in two_genes:
                        joint_prob *= (0.5 * PROBS["mutation"] + 0.5 * (1 - PROBS["mutation"]))
                        # 1st case: the gene copy that the father passes down has to mutate
                        # 2nd case: the gene copy that the father passes down doesn't mutate
                    else:
                        joint_prob *= (0.5 * (1 - PROBS["mutation"]) + 0.5 * (PROBS["mutation"]))
                        # 1st case: father doesn't have any gene copies, so just have to make sure it doesn't mutate
                        # 2nd case: father's gene copy needs to mutate
                elif mother in two_genes:
                    if father in one_gene:
                        joint_prob *= ((1 - PROBS["mutation"]) * 0.5 + PROBS["mutation"] * 0.5)
                    elif father in two_genes:
                        joint_prob *= ((1 - PROBS["mutation"]) * PROBS["mutation"] +
                                       PROBS["mutation"] * (1 - PROBS["mutation"]))
                        # one of the parents have to pass down a mutated gene copy
                    else:
                        joint_prob *= ((1 - PROBS["mutation"]) * (1 - PROBS["mutation"] +
                                                                  PROBS["mutation"] * PROBS["mutation"]))
                        # either both of their gene copies need to mutate or neither of their gene copies mutates
                else:
                    if father in one_gene:
                        joint_prob *= (PROBS["mutation"] * 0.5 + (1 - PROBS["mutation"]) * 0.5)
                    elif father in two_genes:
                        joint_prob *= (PROBS["mutation"] * PROBS["mutation"] +
                                       (1 - PROBS["mutation"]) * (1 - PROBS["mutation"]))
                    else:
                        joint_prob *= (PROBS["mutation"] * (1 - PROBS["mutation"]) +
                                       (1 - PROBS["mutation"]) * PROBS["mutation"])
            elif person in two_genes:
                # both mother and father need to pass down a gene copy
                if person in have_trait:
                    joint_prob *= PROBS["trait"][2][True]
                else:
                    joint_prob *= PROBS["trait"][2][False]
                if mother in one_gene:
                    if father in one_gene:
                        joint_prob *= 0.5 * 0.5
                        # see above logic
                    elif father in two_genes:
                        joint_prob *= 0.5 * (1 - PROBS["mutation"])
                        # making sure father's gene copy doesn't mutate
                    else:
                        joint_prob *= 0.5 * PROBS["mutation"]
                        # father's gene copy needs to mutate
                elif mother in two_genes:
                    if father in one_gene:
                        joint_prob *= (1 - PROBS["mutation"]) * 0.5
                    elif father in two_genes:
                        joint_prob *= (1 - PROBS["mutation"]) * (1 - PROBS["mutation"])
                    else:
                        joint_prob *= (1 - PROBS["mutation"]) * PROBS["mutation"]
                else:
                    if father in one_gene:
                        joint_prob *= PROBS["mutation"] * 0.5
                    elif father in two_genes:
                        joint_prob *= PROBS["mutation"] * (1 - PROBS["mutation"])
                    else:
                        joint_prob *= PROBS["mutation"] * PROBS["mutation"]
            else:
                # both the mother and father have to not pass down the proper gene copy
                if person in have_trait:
                    joint_prob *= PROBS["trait"][0][True]
                else:
                    joint_prob *= PROBS["trait"][0][False]
                if mother in one_gene:
                    if father in one_gene:
                        joint_prob *= 0.5 * 0.5
                    elif father in two_genes:
                        joint_prob *= 0.5 * PROBS["mutation"]
                    else:
                        joint_prob *= 0.5 * (1 - PROBS["mutation"])
                elif mother in two_genes:
                    if father in one_gene:
                        joint_prob *= PROBS["mutation"] * 0.5
                    elif father in two_genes:
                        joint_prob *= PROBS["mutation"] * PROBS["mutation"]
                    else:
                        joint_prob *= PROBS["mutation"] * (1 - PROBS["mutation"])
                else:
                    if father in one_gene:
                        joint_prob *= (1 - PROBS["mutation"]) * 0.5
                    elif father in two_genes:
                        joint_prob *= (1 - PROBS["mutation"]) * PROBS["mutation"]
                    else:
                        joint_prob *= (1 - PROBS["mutation"]) * (1 - PROBS["mutation"])
    if flag:
        joint_prob = 0
    return joint_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities.keys():
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p
        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities.keys():

        # normalizing gene copy probabilities
        sum = 0
        for i in range(3):
            sum += probabilities[person]["gene"][i]
        ratio = 1 / sum
        for i in range(3):
            probabilities[person]["gene"][i] *= ratio

        # normalizing trait probabilities
        sum = probabilities[person]["trait"][True] + probabilities[person]["trait"][False]
        ratio = 1 / sum
        probabilities[person]["trait"][True] *= ratio
        probabilities[person]["trait"][False] *= ratio


if __name__ == "__main__":
    main()
