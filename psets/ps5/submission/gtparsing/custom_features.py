from dependency_features import DependencyFeatures
from numpy import sign

class LexFeats(DependencyFeatures):
    def create_arc_features(self,instance,h,m,add=False):
        """ Notes about the code
        - You start by calling the same function, using the parent class. 
          You can build a chain of feature functions in this way.
        - h provides the index of the head word of the dependency arc
        - m provides the index of the modifier word of the dependency arc
        - You can access the part of speech tags in the instance as instance.pos[i], 
          where i indexes any word token.
        - You can access the words themselves as instance.words[i], 
          where i again indexes the token
        - To create a feature, you call getF(), with two arguments:
          - A feature tuple, which includes an index k, and any other information 
            you want -- it need not be a tuple of exactly three items
          - An argument "add", which you don't need to worry about 
            (but you do need to include)
          - Make sure to keep k up-to-date. 
            This prevents collisions in the space of features.
        """
        ff = super(LexFeats,self).create_arc_features(instance,h,m,add)
        k = len(ff)
        f = self.getF((k,instance.pos[h],instance.words[m]),add)
        ff.append(f)
        return ff


# For Deliverable 1a
class LexDistFeats(LexFeats):
    # Hide this function
    def create_arc_features(self,instance,h,m,add=False):
        ff = super(LexDistFeats,self).create_arc_features(instance,h,m,add)
        k = len(ff)        
        if abs(h - m) <= 10:            
            f = self.getF((k,h - m),add)
        elif h < m:
            f = self.getF((k,-10),add)
        else:
            f = self.getF((k,10),add)
        ff.append(f)
        return ff

# For Deliverable 1b
class LexDistFeats2(LexDistFeats):
    # Hide this function
    def create_arc_features(self,instance,h,m,add=False):
        ff = super(LexDistFeats2,self).create_arc_features(instance,h,m,add)
        k = len(ff)
        f = self.getF((k, instance.pos[m], instance.words[h]),add)
        ff.append(f)
        return ff

# For Deliverable 1c
class ContextFeats(LexDistFeats2):
    # Hide this function
    def create_arc_features(self,instance,h,m,add=False):
        ff = super(ContextFeats,self).create_arc_features(instance,h,m,add)
        
        if h > 0:
            k = len(ff)
            f = self.getF((k,instance.pos[h], instance.pos[h - 1], instance.pos[m]),add)
            ff.append(f)
        if h < len(instance.pos) - 1:
            k = len(ff)
            f = self.getF((k,instance.pos[h], instance.pos[h + 1], instance.pos[m]),add)
            ff.append(f)
        if m > 0: 
            k = len(ff)
            f = self.getF((k,instance.pos[h], instance.pos[m - 1], instance.pos[m]),add)
            ff.append(f)
        if m < len(instance.pos) - 1:
            k = len(ff)
            f = self.getF((k,instance.pos[h], instance.pos[m + 1], instance.pos[m]),add)
            ff.append(f)        
        if m > 0 and h > 0:
            k = len(ff)
            f = self.getF((k,instance.pos[h], instance.pos[h - 1], instance.pos[m - 1], instance.pos[m]),add)
            ff.append(f)
        if m > 0 and h < len(instance.pos) - 1:
            k = len(ff)
            f = self.getF((k,instance.pos[h], instance.pos[h + 1], instance.pos[m - 1], instance.pos[m]),add)
            ff.append(f)            
        if m < len(instance.pos) - 1 and h < len(instance.pos) - 1:
            k = len(ff)
            f = self.getF((k,instance.pos[h], instance.pos[h + 1], instance.pos[m + 1], instance.pos[m]),add)
            ff.append(f)            
        if m < len(instance.pos) - 1 and h > 0:
            k = len(ff)
            f = self.getF((k,instance.pos[h], instance.pos[h - 1], instance.pos[m + 1], instance.pos[m]),add)
            ff.append(f)            
        return ff

# For Deliverable 2c
class DelexicalizedFeats(DependencyFeatures):
    # Hide this function
    def create_arc_features(self,instance,h,m,add=False):
        ff = super(DelexicalizedFeats,self).create_arc_features(instance,h,m,add)
        k = len(ff)
        f = self.getF((k,instance.pos[h],instance.pos[m]),add)
        ff.append(f)
        
        k = len(ff)        
        if abs(h - m) <= 10:            
            f = self.getF((k,h - m),add)
        elif h < m:
            f = self.getF((k,-10),add)
        else:
            f = self.getF((k,10),add)
        ff.append(f)
        
        return ff
        
class LexicalFeats2e(DelexicalizedFeats):
    # Hide this function
    def create_arc_features(self,instance,h,m,add=False):
        ff = super(LexicalFeats2e,self).create_arc_features(instance,h,m,add)
                       
        k = len(ff)
        f = self.getF((k,instance.pos[h],instance.words[m]),add)
        ff.append(f)
        k = len(ff)
        f = self.getF((k, instance.pos[m], instance.words[h]),add)
        ff.append(f)
        
        return ff
        
class FeatsFor2f(LexicalFeats2e):
    
    def create_arc_features(self,instance,h,m,add=False):
        ff = super(FeatsFor2f,self).create_arc_features(instance,h,m,add)
        
        if h > 0:
            k = len(ff)
            f = self.getF((k,instance.pos[h], instance.pos[h - 1], instance.pos[m]),add)
            ff.append(f)            
        if h < len(instance.pos) - 1:
            k = len(ff)
            f = self.getF((k,instance.pos[h], instance.pos[h + 1], instance.pos[m]),add)
            ff.append(f)
        if m > 0: 
            k = len(ff)
            f = self.getF((k,instance.pos[h], instance.pos[m - 1], instance.pos[m]),add)
            ff.append(f)
        if m < len(instance.pos) - 1:
            k = len(ff)
            f = self.getF((k,instance.pos[h], instance.pos[m + 1], instance.pos[m]),add)
            ff.append(f)
        
        if instance.words[h] in self.word_dict:
            head_word = self.word_dict[instance.words[h]]
        else:
            head_word = ''
        if instance.words[m] in self.word_dict:
            mod_word = self.word_dict[instance.words[m]]
        else:
            mod_word = ''
        
        for i in range(1, 4):
            if i <= len(head_word):
                k = len(ff)
                f = self.getF((k, head_word[0 : i]),add)
                ff.append(f)        
            if i <= len(mod_word):
                k = len(ff)
                f = self.getF((k, mod_word[0 : i]),add)
                ff.append(f)
                                           
        return ff
        
class FrenchFeat(FeatsFor2f):
    
    def create_arc_features(self,instance,h,m,add=False):
        ff = super(FrenchFeat,self).create_arc_features(instance,h,m,add)

        if instance.words[h] in self.word_dict:
            head_word = self.word_dict[instance.words[h]]
        else:
            head_word = ''
        if instance.words[m] in self.word_dict:
            mod_word = self.word_dict[instance.words[m]]
        else:
            mod_word = ''
        
        k = len(ff)
        f = self.getF((k, instance.pos[h], head_word.endswith('e'), instance.pos[m], mod_word.endswith('e')),add)
        ff.append(f)
        
        k = len(ff)
        f = self.getF((k, instance.pos[h], head_word.endswith('s'), instance.pos[m], mod_word.endswith('s')),add)
        ff.append(f)
        
        return ff        