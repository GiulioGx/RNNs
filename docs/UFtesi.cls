%%%%%%%%%%%%%%%%%%%%%%%%%%
% This Class is for the Tesi Style of the University
% of Florence.
% The book LaTeX2e class is used and modified.
% F. Bartolini, DIE, via S. Marta 3, 50139, Firenze
% barto@cosimo.ing.unifi.it
%
%  You need LaTeX2e with the graphics bundle installed.
%  You also need the babel package installed for getting
%      italian hyphenation and names
%
%   Usage:
%       \documentclass[laurea|dottorato,twoside|oneside,10pt|11pt|12pt]{UFtesi}
%   Default:
%       \documentclass[laurea,oneside,12pt]{UFtesi.cls}
%
%   Compulsory commands:
%       \title{......}             Thesis title
%       \author{......}            Thesis author
%       \titolocorso{......}       Laurea or doctorate course name
%       \chair{.......}            First thesis supervisor
%       \numberofmembers{..}       Number of supervisors (at least 1)
%   Only for 'laurea' compulsory commands:
%       \degreeyear{......}        Accademic year of graduation
%   Only for 'dottorato' compulsory commands:
%       \periodo{.....}            Doctorate course whole period
%       \curricul{......}          Disciplinary curriculum
%   Available commands:
%       \othermembers{.....\mediskip\\....} Possible other supervisors
%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{UFtesi}
     [23/2/96 v1.0 Classe Latex per le tesi dell'Universita` di Firenze
       by F. Bartolini, Dip. Ing. Elettronica -  Univ. Firenze]
\newcommand{\@ptoption}{}
\newcommand{\@degree}{}
\newif\if@dottorato
\newcommand{\@tipo}{}
\DeclareOption{10pt}{\renewcommand{\@ptoption}{0pt}}
\DeclareOption{11pt}{\renewcommand{\@ptoption}{1pt}}
\DeclareOption{12pt}{\renewcommand{\@ptoption}{2pt}}
\DeclareOption{dottorato}{\@dottoratotrue \renewcommand{\@degree}{Dottorato}}
\DeclareOption{laurea}{\@dottoratofalse \renewcommand{\@degree}{Laurea}}
\DeclareOption{twoside}{\renewcommand{\@tipo}{twoside}}
\DeclareOption{oneside}{\renewcommand{\@tipo}{oneside}}
\ExecuteOptions{laurea,oneside,12pt}
\ProcessOptions
%
% Load the book class for a4 paper dimension
%
\LoadClass[\@tipo,a4paper,1\@ptoption]{book}
% The babel package is required for italian.
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage[english]{babel}
% The graphicx package is required for inserting the University
% logo in the title page.
\RequirePackage{graphicx}
%
% The following variables are defined for the thesis title page
% The name of the doctorate or laurea course
\def\titolocorso#1{\gdef\@titolocorso{#1}}
% Title of the `curriculum' of the Doctorate course
\def\curricul#1{\gdef\@curricul{#1}}
% The Academic year the degree will be officially conferred
\def\degreeyear#1{\gdef\@degreeyear{#1}}
% The period of the Doctorate studies
\def\periodo#1{\gdef\@periodo{#1}}
% The Doctorate cycle
\def\ciclo#1{\gdef\@ciclo{#1}}
% The name of your committee's chair
\def\chair#1{\gdef\@chair{#1}}
% The names of your other committe members, one per line
\def\othermembers#1{\gdef\@othermembers{#1}}
% The number of committee members, which affects both the
% number of lines and the amount of space between lines
% on the approval page.
\def\@numberofmembers{3}
\def\@approvalspace{.75in}
\def\numberofmembers#1{\gdef\@numberofmembers{#1}
\ifnum \@numberofmembers > 3
  \gdef\@approvalspace{.5in}
\fi}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%   UFtesi   %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\linespacing}{1.5}
\renewcommand{\baselinestretch}{\linespacing}
\newcommand{\rilegatura}{10mm}
\addtolength\oddsidemargin{\rilegatura}
\addtolength\evensidemargin{-\rilegatura}
\setlength{\headsep}{15mm}
% begin single line spacing command
\def\BeginSingle{\def\baselinestretch{1.0}\large\normalsize}
% restore normal line spacing command
\def\EndSingle{\par\def\baselinestretch{\linespacing}\large\normalsize}
%%%%%%%%%%%%%%%%%%%

% The abstract which is not present in the original
% LaTeX2e book class is inserted.
% The abstract definition is taken from the report class.
%
\if@titlepage
  \newenvironment{abstract}{%
      \titlepage
      \null\vfil
      \begin{center}
        \bfseries \abstractname
      \end{center}}
     {\par\vfil\null\endtitlepage}
\else
  \newenvironment{abstract}{%
      \if@twocolumn
        \section*{\abstractname}%
      \else
        \small
        \begin{center}%
          {\bfseries \abstractname\vspace{-.5em}\vspace{\z@}}%
        \end{center}%
        \quotation
      \fi}
      {\if@twocolumn\else\endquotation\fi}
\fi

% The titlepage environment is changed so that
% a 1 \baselinestretch is used.
% The global \baselinestretch (as defined by
% the \linespacing command above) is then restored.
%
\renewenvironment{titlepage}
    {%
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \renewcommand{\baselinestretch}{1.0}
      \thispagestyle{empty}%
      \if@compatibility
        \setcounter{page}{0}
      \else
        \setcounter{page}{1}%
      \fi}%
    {\if@restonecol\twocolumn \else \newpage \fi
      \renewcommand{\baselinestretch}{\linespacing}
      \setcounter{page}{1}%
    }

% The \maketitle command is adapted to build a
% title page suitable for the University of Florence.
%
\renewcommand{\maketitle}
{\begin{titlepage}%
  \let\footnotesize\small
  \let\footnoterule\relax
  \BeginSingle
  \vspace*{-25mm}
  \centerline{\includegraphics[height=40mm]{stemma}}
  \begin{center}
    \textsc{}\medskip\\
    {\Large Scuola di Ingegneria}\medskip\\
    \rule{50mm}{0.01mm}\medskip\\
    \if@dottorato
      {\large Dottorato di Ricerca in}\medskip\\
    \else
      {\large Corso di Laurea in}\medskip\\
    \fi
    \Large\textsc{\@titolocorso}\medskip\\
    \if@dottorato
      \@curricul
    \fi
    \vfill
    {\LARGE \bf \@title \par}
    \bigskip\bigskip\bigskip\bigskip
    \large{Tesi di \@degree~di} \par\bigskip
    {\Large \@author} \par
    \@date \par
  \end{center}\par
  \vfill
  \noindent
  \if@dottorato
    \ifnum \@numberofmembers > 1
      Tutori:
    \else
      Tutore:
    \fi
  \else
    \ifnum \@numberofmembers > 1
      \large\textbf{Relatori}:\ \ 
    \else
      \large\textbf{Relatore}:
    \fi
  \fi
  \begin{quote}
    {\Large \@chair}
  \end{quote}
    \ifnum \@numberofmembers > 1
    \large\textbf{Correlatori}:
    \begin{quote}
      {\Large \@othermembers}
     \fi
  \end{quote}
  \vfill
  \begin{center}
    \rule{40mm}{0.01mm}\\
    \if@dottorato
      {Ciclo \@ciclo}\\
      {Periodo \@periodo}
    \else
      {Anno Accademico \@degreeyear}
    \fi
  \end{center}
  \end{titlepage}
  \setcounter{footnote}{0}%
  \let\thanks\relax\let\maketitle\relax
  \gdef\@thanks{}\gdef\@author{}\gdef\@title{}
  \EndSingle
}

% The page heading are changed. The uppercase forcing
% is avoided and a rule is inserted below the heading.
%
\if@twoside
  \def\ps@headings{%
      \let\@oddfoot\@empty\let\@evenfoot\@empty
      \def\@evenhead{\thepage\hfil\slshape\leftmark%
      \hspace{-\textwidth}\rule[-2mm]{\textwidth}{0.01mm}}%
      \def\@oddhead{{\slshape\rightmark}\hfil\thepage%
      \hspace{-\textwidth}\rule[-2mm]{\textwidth}{0.01mm}}%
      \let\@mkboth\markboth
    \def\chaptermark##1{%
      \markboth {
        \ifnum \c@secnumdepth >\m@ne
          \if@mainmatter
            \@chapapp\ \thechapter. \ %
          \fi
        \fi
        ##1}{}}%
    \def\sectionmark##1{%
      \markright {%
        \ifnum \c@secnumdepth >\z@
          \thesection. \ %
        \fi
        ##1}}}
\else
  \def\ps@headings{%
    \let\@oddfoot\@empty
    \def\@oddhead{{\slshape\rightmark}\hfil\thepage%
    \hspace{-\textwidth}\rule[-2mm]{\textwidth}{0.01mm}}%
    \let\@mkboth\markboth
    \def\chaptermark##1{%
      \markright {%
        \ifnum \c@secnumdepth >\m@ne
          \if@mainmatter
            \@chapapp\ \thechapter. \ %
          \fi
        \fi
        ##1}}}
\fi

% The caption is changed so that a \small font is used.
%
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \BeginSingle
  \sbox\@tempboxa{{\small #1: #2}}%
  \ifdim \wd\@tempboxa >\hsize
    {\small #1: #2}\par
  \else
    \global \@minipagefalse
    \hbox to\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \EndSingle
  \vskip\belowcaptionskip}
%
% The \tableofcontents command is changed so that
% the \uppercase forcing of the header is avoided
%
\renewcommand{\tableofcontents}{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\contentsname
        \@mkboth{\contentsname}{\contentsname}}%
    \@starttoc{toc}%
    \if@restonecol\twocolumn\fi
    }
%
% The \listoffigures command is changed so that
% the \uppercase forcing of the header is avoided
%
\renewcommand{\listoffigures}{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listfigurename
      \@mkboth{\listfigurename}%
              {\listfigurename}}%
    \@starttoc{lof}%
    \if@restonecol\twocolumn\fi
    }
%
% The \listoftables command is changed so that
% the \uppercase forcing of the header is avoided
%
\renewcommand{\listoftables}{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listtablename
      \@mkboth{\listtablename}{\listtablename}}%
    \@starttoc{lot}%
    \if@restonecol\twocolumn\fi
    }
% The thebibliography environment is changed so that
% the bibliography page is inserted in the list of content.
%
%\renewenvironment{thebibliography}[1]
%     {\chapter*{\bibname
%        \@mkboth{\bibname}{\bibname}}%
%        \addcontentsline{toc}{chapter}{\bibname}%
%      \list{\@biblabel{\arabic{enumiv}}}%
%           {\settowidth\labelwidth{\@biblabel{#1}}%
%            \leftmargin\labelwidth
%            \advance\leftmargin\labelsep
%            \if@openbib
%              \advance\leftmargin\bibindent
%              \itemindent -\bibindent
%              \listparindent \itemindent
%              \parsep \z@
%            \fi
%            \usecounter{enumiv}%
%            \let\p@enumiv\@empty
%            \renewcommand{\theenumiv}{\arabic{enumiv}}}%
%      \BeginSingle
%      \if@openbib
%        \renewcommand{\newblock}{\par}
%      \else
%        \renewcommand{\newblock}{\hskip .11em \@plus.33em \@minus.07em}%
%      \fi
%      \sloppy\clubpenalty4000\widowpenalty4000%
%      \sfcode`\.=\@m}
%     {\def\@noitemerr
%       {\@latex@warning{Empty `thebibliography' environment}}%
%      \EndSingle
%      \endlist}
\renewenvironment{thebibliography}[1]
     {\chapter*{\bibname
        \@mkboth{\bibname}{\bibname}}%
        \addcontentsline{toc}{chapter}{\bibname}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \BeginSingle
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \EndSingle
      \endlist}

% The \part command is changed so that the page number is
% not inserted in the part starting page.
%
\renewcommand{\part}{\cleardoublepage
                 \thispagestyle{empty}%
                 \if@twocolumn
                     \onecolumn
                     \@tempswatrue
                   \else
                     \@tempswafalse
                 \fi
                 \hbox{}\vfil
                 \secdef\@part\@spart}
%
% set single spacing in table environment
%\renewenvironment{table}
%               {\BeginSingle\@float{table}}
%               {\EndSingle\end@float}
%\renewenvironment{table*}
%               {\BeginSingle\@dblfloat{table}}
%               {\EndSingle\end@dblfloat}
