import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for var in self.domains.keys():
            to_remove = []
            for word in self.domains[var]:
                if len(word) != var.length:
                    to_remove.append(word)
            for word in to_remove:
                self.domains[var].remove(word)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        if self.crossword.overlaps[x, y] is None:
            return False
        i = self.crossword.overlaps[x, y][0]
        j = self.crossword.overlaps[x, y][1]
        to_remove = set()
        for word_x in self.domains[x]:
            flag = False
            for word_y in self.domains[y]:
                if word_x[i: i + 1] == word_y[j: j + 1]:
                    flag = True
            # If no value in y's domain satisfied word_x
            if not flag:
                to_remove.add(word_x)
                revised = True
        self.domains[x] = self.domains[x] - to_remove
        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs is None:
            queue = [(v1, v2) for v1 in self.crossword.variables for v2 in self.crossword.neighbors(v1)]
        else:
            queue = arcs
        while queue:
            arc = queue[0]
            if self.revise(arc[0], arc[1]):
                """if the domain of one of the variables has been revised, we need to also check the arcs with that 
                    variable and all of its other neighbors, while disregarding the neighbor we just checked"""
                for v2 in self.crossword.neighbors(arc[0]) - {arc[1]}:
                    queue.append((v2, arc[0]))
            if len(self.domains[arc[0]]) == 0:
                return False
            queue.remove(arc)
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        if len(assignment.keys()) == len(self.crossword.variables):
            return True
        return False

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        for var1 in assignment.keys():
            # making sure that each variable meets unary constraints
            if len(assignment[var1]) != var1.length:
                return False
            for var2 in assignment.keys():
                if var1 == var2:
                    continue
                # making sure all values are distinct
                if assignment[var1] == assignment[var2]:
                    return False
                # enforcing arc consistency with new values
                if self.crossword.overlaps[var1, var2] is not None:
                    i = self.crossword.overlaps[var1, var2][0]
                    j = self.crossword.overlaps[var1, var2][1]
                    if assignment[var1][i: i + 1] != assignment[var2][j: j + 1]:
                        return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        value_weights = {word: 0 for word in self.domains[var]}
        for word_x in value_weights.keys():
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment.keys():
                    if len(assignment[neighbor]) == 1:
                        continue
                i = self.crossword.overlaps[var, neighbor][0]
                j = self.crossword.overlaps[var, neighbor][1]
                for word_y in self.domains[neighbor]:
                    if word_x[i: i + 1] != word_y[j: j + 1] or word_x == word_y:
                        value_weights[word_x] += 1
                        # increase weight of word if rule out more values in the neighbors of var
        sorted_values = sorted(value_weights.items(), key=lambda x: x[1])
        # sorted in ascending order
        values = [value[0] for value in sorted_values]
        return values

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        min_num = -1
        for var in self.crossword.variables:
            if var in assignment.keys():
                continue
            # initial value
            if min_num == -1:
                chosen_var = var
                min_num = len(self.domains[var])
                continue
            # comparing number of remaining values
            if len(self.domains[var]) < min_num:
                min_num = len(self.domains[var])
                chosen_var = var
            if len(self.domains[var]) == min_num:
                # comparing degree
                if len(self.crossword.neighbors(var)) > len(self.crossword.neighbors(var)):
                    chosen_var = var
        return chosen_var

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment
        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            new_assignment = assignment.copy()
            new_assignment.update({var: value})
            if self.consistent(new_assignment):
                result = self.backtrack(new_assignment)
                if result is not None:
                    return result
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
