def collect_input(self, f, indices):
    '''Collect from the input section of the file.'''

    # Create an upper-case and stripped duplicate of the input block.
    # Remove comments (:: or !).
    ei = indices['INPUT END']
    f = list(f)
    search = f[:ei]
    for i, x in enumerate(search):
        search[i] = x.upper().lstrip()
        search[i] = search[i].partition('::')[0]
        search[i] = search[i].partition('!')[0]
        search[i] = search[i].partition('#')[0]
        f[i] = f[i].partition('::')[0]
        f[i] = f[i].partition('!')[0]
        f[i] = f[i].partition('#')[0]
    f = tuple(f)

    # Let's first look for block-type keywords.  The block-type has the
    # keyword followed by a parameter block terminated with 'END'.
    # We'll try each known block-type keyword, and if it located find
    # the END keyword and collect.  Otherwise, we'll move to the next
    # keyword.
    en = enumerate
    for keyword in self.blockkeys:
        s = next((i for i, x in en(search) if x == keyword), -1)
        if s == -1:
            pass
        else:
            s += 1
            # Locate END for this block
            e = next(i for i, x in en(search[s:], s) if x == 'END')
            # Make the parameters a tuple in this key
            self.key[keyword] = tuple(x for x in f[s:e])

    # Now, lets locate the line-type keywords. The line-type has the
    # keyword and parameters on the same line.
    for keyword in self.linekeys:
        l = len(keyword)
        ar = [i for i, x in en(search) if keyword == x[:l]]
        for ix in ar:
            # Split this line, then add parameters after keyword to key
            line = ' '.join(f[ix].lstrip().split()[1:])
            # Can have multiple instances of the keyword
            try:
                self.key[keyword].append(line)
            except KeyError:
                self.key[keyword] = []
                self.key[keyword].append(line)

    # Last, let's find the single-type keywords.  The single-type
    # is just a keyword with no parameters.
    for keyword in self.singlekeys:
        ix = next((i for i, x in en(search) if x == keyword), -1)
        if ix == -1:
            pass
        else:
            self.key[keyword] = True
