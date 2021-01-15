.. title: Dansk Resumé
.. slug: dansk-resume
.. date: 2021-01-15 12:58:53 UTC
.. tags: 
.. category: 
.. link: 
.. description: 
.. type: text

Maskinlæring (’machine learning’) er et redskab til modellering og
analyse af geovidenskabelige data Jeg har sat den seneste udvikling
inden for dyb læring (’deep learning’) ind i en større sammenhæng
indenfor maskinlæring ved at gennemlæse de tilgange og udfordringer som
maskinlæring har inden for geovidenskab. Afhandlingen består af seks
peer-reviewed udgivelser og en indsendt journalartikel. Yderligere er
der fem peer-reviewed udgivelser i appendix.

Formålet med denne afhandling er, at anvende den seneste udvikling inden
for systemer for computer vision, neurale netværk og maskinlæring for
geovidenskabelige data, især 4D seismisk analyse. Neurale netværk er en
type maskinlæring, der har bidraget betydeligt til moderne kunstig
intelligens og automatisering. Det blev på et tidligt tidspunkt
anerkendt inden for geofysik, at neurale netværk var anvendelige. Brugen
af neurale netværk for deres evne til at være universelle
funktions-approksimatorer blev tidligt anderkendt inden for geofysik.
Grundet den nylige interesse for dyb læring, har neurale netværk oplevet
en renæssance inden for geovidenskabelige anvendelser, særligt
automatisk seismisk fortolkning, inverterings processor og
sekvensmodellering.

Dette efterfølges af en udforskning af uovervåget læring til segmentring
af kalksedimenter i tilbagesprednings-elektronmikroskopi ”back-scatter
scanning electron microscopy” data. Det næste kapitel viser, at brugen
af neurale netværk prætrænede på billeder, kan reducere den nødvendige
mængde data, der er nødvendige for at overføre læring til
geovidenskabelige problemer. Kapitlet derefter viser, at foldninger med
komplekse tal kan stabilisere træningen og datakompressionen af
ikke-stationære fysiske data. Derpå beregnes tryk og mætningsdata med
brug af 4D seismiske data ved hjælp af et nyt dybt tæt prøvebaseret
indkoder-dekoder netværk. Netværket indeholder et fysisk grundlag, for
selv at lære resten af inversionsprocessen. Arbejdet viser overførsel
fra simulerede til rigtige data er muligt.

Endelig blev der udviklet en uovervåget ’unsupervised’ metode, til at
udregne 3D-tidsforskydninger fra to 4D seismiske kuber. Netværket
beregner disse 3D tidsskift inklusiv usikkerhedsmålinger på dem. På
grund af de beregningsmæssige omkostninger og dårlig kvalitet, bliver
disse normalt kun beregnet i 1D. Inden for træningsløkken integreres det
stationære hastighedsfelt numerisk for at få 3D tidsskift, som er
begrænset af topologien på en geologisk konsistent måde. Den
uovervågende implementation af netværksstrukturen sikrer at bias fra
andre tidsforskydnings ekstraktionsmetoder ikke implicit indgår i
netværket. Den uovervågende metode lærer netværket at følge en bestemt
opførsel uden brug af sande ”ground truth” eksempler. Yderligere,
styrker dette tilliden til systemet, da ekstrationsmetoden begrænses til
det dybe læringssystem og veldefinerede oprationer inden for dette som
automatiserer den uovervågede træning.