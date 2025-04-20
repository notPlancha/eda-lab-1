#let to = box(width: 8pt, baseline: 10%)[..]
// 
// #let grau = {
//   $#h(-1pt)<#h(0pt)#text(baseline: 0.5pt)[k]#h(0pt)>#h(1pt)$
// }
// #let grau2 = {
//   $#h(-1pt)<#h(0pt)#text(baseline: 0.5pt)[k]^2#h(0pt)>#h(1pt)$
// }
#let project(
  institution: "Institution",
  below_institution: "Some Other Information",
  title: "Hello World",
  subtitle: "The text below the title",
  abstract: [],
  authors: (),
  date: none,
  lang: "en",
  page_numbering: "1",
  // Typst's default, really close to Times New Roman
  text_font: "libertinus serif",
  heading_font: "libertinus serif",
  reset_page: false,
  font-size: 12pt,
  bibliography_call: bibliography("refs.bib", title: "References"),
  after_date: none,
  body,
) = {
  set document(author: authors.map(a => a.name), title: title)
  set text(lang: lang, font: text_font, size: font-size)
  show heading: set text(font: heading_font)

  page(margin: (
    top:50pt,bottom: 130pt, x:100%-86%
  ), par(justify: false, align(center + horizon, 
    grid(columns: 1, rows: (1fr, 3fr, 6fr, 0.7fr), 
      [
        #show text: smallcaps
        #set par(leading: 3pt)
        #text(13pt, institution) \ #text(12.5pt, below_institution)
      ],
      [ 
        #set par(leading: 9.5pt)
        #line(length: 100%, stroke: 0.5pt) \
        #v(-35pt)
        #text(28pt, title)
        #set par(leading: 5pt)
        #text(16pt, subtitle)
        #line(length: 100%, stroke: 1pt)
      ],
      [
        #set par(leading: 8.5pt)
        #for author in authors {
          [
            #text(20pt, author.name) \
            #if "email" in author [#link("mailto:" + author.email)] else [#none]
            #v(8pt)
          ]
        }
      ],
      [
        #set par(leading: 0.2cm)
        #set text(16pt)
        #date \ #after_date
      ]
      
  ))))
  pagebreak()
  
  
  if reset_page {counter(page).update(1)}
  set page(numbering: page_numbering, number-align: center)
  set heading(bookmarked: true)
  set raw(tab-size: 2)
  show raw: set text(
    font: "Fira Code"
  )
  show raw.where(block:true): it => [
    #block(fill: luma(230), inset: 6pt, radius: 1pt, it)
  ]
  set text(hyphenate: true)
  set par(justify: true)
  //set math.equation(numbering: "(1)") // broken fsr
  /*
  show ref: it => {
    let eq = math.equation
    let el = it.element
    if el != none and el.func() == eq {
      numbering(
        el.numbering,
        ..counter(eq).at(el.location())
      )
    } else {
      it
    }
  }
  */
  show figure.where(
    kind: table
  ): set figure.caption(position: top)
  // show figure.where(
  //   kind: algo
  // ): set figure(supplement: "Algoritmo", placement: none)
  // show figure.where(
  //   kind: algo
  // ): set block(breakable: true)
  set list(indent: 0.6cm)

  [
    #body
    #bibliography_call
  ]
}