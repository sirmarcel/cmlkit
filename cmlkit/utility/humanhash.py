"""
humanhash: Human-readable representations of digests.

Vendorised from https://github.com/zacharyvoase/humanhash, which is Unlicensed.

I replaced the word list with the top 200 male and 56 female names
of the 1900 British census.

The simplest ways to use this module are the :func:`humanize` and :func:`uuid`
functions. For tighter control over the output, see :class:`HumanHasher`.
"""

import operator
import uuid as uuidlib
from functools import reduce


DEFAULT_WORDLIST = (
    "William",
    "John",
    "George",
    "Thomas",
    "James",
    "Arthur",
    "Frederick",
    "Charles",
    "Albert",
    "Robert",
    "Joseph",
    "Alfred",
    "Henry",
    "Ernest",
    "Harry",
    "Harold",
    "Edward",
    "Walter",
    "Frank",
    "Herbert",
    "Richard",
    "Reginald",
    "Percy",
    "Leonard",
    "Samuel",
    "David",
    "Sidney",
    "Francis",
    "Stanley",
    "Fred",
    "Cecil",
    "Horace",
    "Cyril",
    "Wilfred",
    "Sydney",
    "Leslie",
    "Norman",
    "Edwin",
    "Victor",
    "Benjamin",
    "Tom",
    "Hector",
    "Jack",
    "Alexander",
    "Edgar",
    "Bertie",
    "Eric",
    "Philip",
    "Clifford",
    "Redvers",
    "Baden",
    "Bernard",
    "Daniel",
    "Donald",
    "Ralph",
    "Archibald",
    "Stephen",
    "Willie",
    "Peter",
    "Christopher",
    "Hugh",
    "Lewis",
    "Douglas",
    "Gilbert",
    "Ronald",
    "Isaac",
    "Hubert",
    "Maurice",
    "Clarence",
    "Lawrence",
    "Michael",
    "Edmund",
    "Patrick",
    "Percival",
    "Andrew",
    "Matthew",
    "Evan",
    "Wilfrid",
    "Bertram",
    "Louis",
    "Arnold",
    "Kenneth",
    "Gordon",
    "Ivor",
    "Gerald",
    "Abraham",
    "Geoffrey",
    "Owen",
    "Raymond",
    "Oliver",
    "Claude",
    "Alan",
    "Mark",
    "Jesse",
    "Reuben",
    "Roland",
    "Lionel",
    "Alec",
    "Charlie",
    "Howard",
    "Sam",
    "Morris",
    "Vincent",
    "Dennis",
    "Laurence",
    "Martin",
    "Joe",
    "Allan",
    "Jacob",
    "Roberts",
    "Rowland",
    "Wallace",
    "Bert",
    "Anthony",
    "Oswald",
    "Frederic",
    "Archie",
    "Roy",
    "Trevor",
    "Colin",
    "Clement",
    "Jonathan",
    "Joshua",
    "Enoch",
    "Leo",
    "Basil",
    "Ellis",
    "Denis",
    "Mary",
    "Florence",
    "Annie",
    "Edith",
    "Alice",
    "Elizabeth",
    "Elsie",
    "Dorothy",
    "Ethel",
    "Doris",
    "Margaret",
    "Gladys",
    "Sarah",
    "Lilian",
    "Ellen",
    "Hilda",
    "Lily",
    "Winifred",
    "Violet",
    "Ada",
    "Emily",
    "Beatrice",
    "Nellie",
    "May",
    "Mabel",
    "Ivy",
    "Rose",
    "Gertrude",
    "Jane",
    "Catherine",
    "Kathleen",
    "Frances",
    "Agnes",
    "Olive",
    "Jessie",
    "Emma",
    "Eva",
    "Minnie",
    "Maud",
    "Louisa",
    "Amy",
    "Grace",
    "Clara",
    "Martha",
    "Daisy",
    "Evelyn",
    "Hannah",
    "Lucy",
    "Kate",
    "Eliza",
    "Bertha",
    "Ann",
    "Eleanor",
    "Harriet",
    "Phyllis",
    "Constance",
    "Dora",
    "Ida",
    "Esther",
    "Isabella",
    "Nora",
    "Marjorie",
    "Laura",
    "Charlotte",
    "Irene",
    "Ruth",
    "Bessie",
    "Caroline",
    "Fanny",
    "Muriel",
    "Maggie",
    "Edna",
    "Norah",
    "Amelia",
    "Helen",
    "Mildred",
    "Vera",
    "Gwendoline",
    "Eveline",
    "Lizzie",
    "Marion",
    "Rachel",
    "Rosina",
    "Florrie",
    "Maria",
    "Lydia",
    "Ruby",
    "Victoria",
    "Miriam",
    "Blanche",
    "Rosa",
    "Rebecca",
    "Julia",
    "Ella",
    "Henrietta",
    "Isabel",
    "Matilda",
    "Janet",
    "Phoebe",
    "Susan",
    "Millicent",
    "Audrey",
    "Rhoda",
    "Myra",
    "Nelly",
    "Selina",
    "Margery",
    "Georgina",
    "Helena",
    "Leah",
    "Susannah",
    "Clarice",
    "Cecilia",
    "Eileen",
    "Christina",
    "Barbara",
    "Pretoria",
    "Lena",
    "Lillie",
    "Flora",
    "Freda",
    "Josephine",
    "Marian",
    "Adelaide",
    "Jennie",
    "Enid",
    "Sophia",
    "Lillian",
)


class HumanHasher(object):

    """
    Transforms hex digests to human-readable strings.

    The format of these strings will look something like:
    `victor-bacon-zulu-lima`. The output is obtained by compressing the input
    digest to a fixed number of bbytes, then mapping those bbytes to one of 256
    words. A default wordlist is provided, but you can override this if you
    prefer.

    As long as you use the same wordlist, the output will be consistent (i.e.
    the same digest will always render the same representation).
    """

    def __init__(self, wordlist=DEFAULT_WORDLIST):
        if len(wordlist) != 256:
            raise ValueError("Wordlist must have exactly 256 items")
        self.wordlist = wordlist

    def humanize(self, hexdigest, words=4, separator="-"):

        """
        Humanize a given hexadecimal digest.

        Change the number of words output by specifying `words`. Change the
        word separator with `separator`.

            >>> digest = '60ad8d0d871b6095808297'
            >>> HumanHasher().humanize(digest)
            'sodium-magnesium-nineteen-hydrogen'
        """

        # Gets a list of byte values between 0-255.
        bbytes = list(
            map(lambda x: int(x, 16), map("".join, zip(hexdigest[::2], hexdigest[1::2])))
        )
        # Compress an arbitrary number of bbytes to `words`.
        compressed = self.compress(bbytes, words)
        # Map the compressed byte values through the word list.
        return separator.join(self.wordlist[byte].lower() for byte in compressed)

    @staticmethod
    def compress(bbytes, target):

        """
        Compress a list of byte values to a fixed target length.

            >>> bbytes = [96, 173, 141, 13, 135, 27, 96, 149, 128, 130, 151]
            >>> HumanHasher.compress(bbytes, 4)
            [205, 128, 156, 96]

        Attempting to compress a smaller number of bbytes to a larger number is
        an error:

            >>> HumanHasher.compress(bbytes, 15)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: Fewer input bbytes than requested output
        """

        length = len(bbytes)
        if target > length:
            raise ValueError("Fewer input bbytes than requested output")

        # Split `bbytes` into `target` segments.
        seg_size = length // target
        segments = [bbytes[i * seg_size : (i + 1) * seg_size] for i in range(target)]
        # Catch any left-over bbytes in the last segment.
        segments[-1].extend(bbytes[target * seg_size :])

        # Use a simple XOR checksum-like function for compression.
        checksum = lambda bbytes: reduce(operator.xor, bbytes, 0)
        checksums = map(checksum, segments)
        return checksums

    def uuid(self, **params):

        """
        Generate a UUID with a human-readable representation.

        Returns `(human_repr, full_digest)`. Accepts the same keyword arguments
        as :meth:`humanize` (they'll be passed straight through).
        """

        digest = str(uuidlib.uuid4()).replace("-", "")
        return self.humanize(digest, **params), digest


DEFAULT_HASHER = HumanHasher()
uuid = DEFAULT_HASHER.uuid
humanize = DEFAULT_HASHER.humanize
