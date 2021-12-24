from os import system
from query import Query
from document import Document
import time
import tracemalloc

sizeLimitWord = 0
arrayOpc = []
listStopWords = ""
limitPostings = 100000
currentPostings = 0
nameFileTsvGz = "amazon_reviews_us_Digital_Music_Purchase_v1_00.tsv.gz"
finalFileNames = ["'-1", "1'-1945", '19456199-2', "2'-28", "28'-34", "34'sec-5", "5'-80", "80'-abscur", 'abseit-acoust', 'acoustason-add', "add'-age", "age'-album", "album'-along", "along'-also", "also'-alway", "alway'-amaz", 'amaza-american', "american'-anoint", 'anointd-anymor', "anymore'-appeal", "appeal'-around", "around'-artist", "artist'-attent", 'attenti-awar', 'awarapan-awesom', 'awesomaz-back', "back'-ballad", "ballad'-band", "band'-bassnectar", "bassnectar'-beatl", "beatle'-beauti", 'beautician-believ', "believ'n-best", "best'-beyond", "beyond'-bit", "bit'-blood", "blood'-bold", "bold'-bought", "bought'-br", "br'-brilliant", "brilliant'-brought", 'broughta-bust', "bust'-buy", "buy'-came", "came'-candl", "candle'-carey", "carey'-catchi", 'catchier-cd', "cd'-chanc", "chance'-check", "check'-choos", 'choosen-christma', "christma'-classic", "classic'-close", "close'-collect", 'collectablb-come', "come'-complet", 'completa-confid', 'confidant-continu', "continua-cosby'", 'cosbyism-countri', 'countribean-crash', "crash'-crosbi", "crosby'-cuz", "cuz'-danc", 'danc3-day', "day'-decid", 'decida-definit', 'definital-depech', "depeche'-devot", "devot'd-dig", "dig'-disappoint", "disappoint'-dj", "dj'-dope", "dope'-download", "download'-driven", "driven'-dvd", "dvd'-easi", 'easier-effect', "effect'-els", 'elsa-end', "end'-enjoy", "enjoy'-ep", "ep'-esso", "esso'-even", 'evenaft-everi', 'everibodi-exactli', 'exactlli-excit', 'excited-explicit', "explicit'-face", "face'-famili", 'familia-fan', "fan'-fast", "fast'-favorit", 'favorita-featur', "feature'-fela", "fela'-fillmor", "fillmore'-find", "find'-first", "first'-five", "five'-focus", "focus'-forgiv", 'forgiven-found', "found'-free", "free'-fulfil", 'fulfild-funer', 'funera-game', "game'-genr", 'genra-get', "get'-give", "give'-go", "go'-gone", "gone'-good", "good'-got", "got'-great", "great'-groundbreak", 'groundbreakingli-guilti', 'guiltier-guy', "guy'-happen", "happen'-harlem", "harlem'-he'", "he''-hear", "hear'-heard", "heard'-heavi", 'heavier-hey', "hey'-highwood", 'highyl-hit', "hit'-honest", "honest'-hopeless", 'hopelessi-howev', "howeve-i'd", "i'd'v-i'm", "i'm'-idol", "idol'-impress", 'impressario-individu', 'individual-inspir', "inspir'd-interest", "interest'-isbel", "isbel'-jame", "jame'-jesu", "jesu'-jolli", 'jollier-k', "k'-keke", "keke'-killer", "killer'-knew", "knew'-knowledg", "knowledge'-ladi", 'ladi6-late', "late'-learn", 'learnabl-lerch', "lerche'-licens", "license'-light", "light'-like", "like'-listen", "listen'-live", "live'-long", "long'-lord", "lord'-loud", "loud'-love", "love'-lyric", "lyric'-mahanttan", 'mahanu-make', "make'-mani", "mani'-master", "master'-may", "may'-medit", 'meditacao-memor', 'memorab-messia', 'messiaah-might', "might'v-minut", 'minuta-mix', "mix'-monster", "monster'-motion", "motion'-movi", 'movia-much', "much'-music", "music'-myspac", "myspam-nature'", "naturebot'-neighborhood", "neighborhood'-never", "never'-new", "new'-nice", "nice'-non", 'nona-notion', 'notiv-ode', "ode'-old", "old'-one", "one'-orchestra", "orchestra'-origin", "origin'-overcom", 'overcomethi-pandora', "pandora'-particular", "particular'-pay", "pay'-pere", "pere'-perform", 'performa-philip', "philip'-pie", "pie'-plain", "plain'-play", "play'-pluck", "pluck'-pop", "pop'-power", "power'-pretend", 'pretenda-princ', 'princapl-product', "product'-psych", "psych'-purchas", 'purchasaew-qualiti', 'qualitit-r', "r'-rambl", 'rambla-rave', 'raveboy-realiti', "realities'-realli", 'reallif-recommend', 'recommenda-recov', 'recoveri-rejuven', 'rejuvin-releas', 'releasd-remind', "remind'-represent", "represent'-reveal", "revealed'-rich", "rich'-ring", "ring'-rock", "rock'-ross", "ross'-sad", "sad'-sane", "sane'-say", "say'-sea", "sea'-see", "see'-send", "send'-set", "set'-shatter", "shatter'-show", "show'-simpl", 'simplament-sing', "sing'-singl", 'singlabl-sleep', "sleep'-smooth", "smooth'-solstic", 'solstiti-sometim', "sometime'-song", "song'-soul", "soul'-sound", "sound'-speak", "speak'-spontan", 'spontana-star', "star'-steam", "steam'-still", "still'-straight", "straight'-stubborn", 'stubbornli-style', "style'-summon", "summoner'-sure", "sure'-switchfoot", "switchfoot'-take", "take'-tap", "tap'-telecast'", 'telecasteri-test', "test'-that'", "that''-thing", "thing'-thoroughli", 'thoroughlygood-three', "three'-time", "time'-toe", "toe'-tool", "tool'-tow", 'towa-track', "track'-trend", 'trenda-true', "true'-tune", "tune'-two", "two'-underr", 'underrart-unlik', 'unlikeli-us', "us'-v2", 'v2000-version', "version'-violin", "violin'-vocalist", "vocalist'-voic", 'voical-wait', "wait'-want", "want'-way", "way'-well", "well'-whitney", "whitney'-william", "william'-without", "without'-wonder", "wonder'-work", "work'-worth", "worth'-would", "would'a-wrong", "wrong'-year", "year'-younger", "younger'-ﾟ"]
      
def options():
        """ Insert a dataSaves in an array the various options chosen by the user to be used several times later """
        global sizeLimitWord
        global arrayOpc
        global listStopWords
        
        if len(arrayOpc) != 0:
            return
        
        method = str(input("Choose the option:\n" +
                           "1 -> Minimum length\n" +
                           "2 -> Skip\n"))    
        if method == '1':
                sizeLimitWord = int(input("Enter size:\n"))
                
        arrayOpc.append(method)
        
        method = str(input("Choose the option:\n" +
                           "1 -> Skip\n" +
                           "2 -> Default list (NLTK)\n" +
                           "3 -> User Defined\n"))
        
        if method == '3':
                listStopWords = str(
                    input("Enter words separated by spaces to serve as stop words:\n"))
                
        arrayOpc.append(method)
        
        method = str(input("Choose the option:\n" +
                           "1 -> Skip\n" +
                           "2 -> Default list (NLTK)\n" +
                           "3 -> Snowball\n"))
        arrayOpc.append(method)
    
def run():
    objective = str(input("Choose the option:\n" +
                             "1 -> Index\n" +
                             "2 -> Search\n" +
                             "3 -> Analysis\n" +
                             "4 -> Exit\n"))             
    if objective == "1":
        options()
        doc = Document(nameFileTsvGz, limitPostings, currentPostings, arrayOpc, sizeLimitWord, listStopWords, finalFileNames)
        t0 = time.time()
        doc.indexar()
        t1 = time.time()
        print("Time it takes to do the indexing: " + str(t1 - t0) + "s")
        doc.weightNormalized()
    elif objective == "2":
        options()
        query = Query(nameFileTsvGz, limitPostings, currentPostings, arrayOpc, sizeLimitWord, listStopWords)
        t0 = time.time()
        query.query()
        t1 = time.time()
        print("Time it takes to do the search: " + str(t1 - t0) + "s")
    elif objective == "3":
        options()
        query = Query(nameFileTsvGz, limitPostings, currentPostings, arrayOpc, sizeLimitWord, listStopWords)
        query.analyses()
    elif objective == '4':
         system.exit(4)

tracemalloc.start()
run()
print("Memory estimate used:"+str(tracemalloc.get_traced_memory()[1]-tracemalloc.get_traced_memory()[0]) + "bytes")
tracemalloc.stop()