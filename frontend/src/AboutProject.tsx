
import { Volume2 } from 'lucide-react'

export default function AboutProject() {
  return (
    <section className="project-banner" id="about">
      <div className="banner-orb">
        <Volume2 size={24} />
      </div>

      <div className="banner-copy">
        <span>THE STORY BEHIND DARIJAFLOW</span>

        <h2>La technologie rencontre la culture.</h2>

        <p>
          Un projet de recherche et d’expérimentation autour de la
          traduction anglais–darija.
        </p>
      </div>

      <div className="banner-mark">
        د<span>.</span>
      </div>
    </section>
  )
}
