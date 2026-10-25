
import { Languages, Mic2, WandSparkles } from 'lucide-react'

export default function HowItWorks() {
  return (
    <section className="features-section" id="how-it-works">
      <div className="feature-intro">
        <div className="section-kicker">
          CONÇU AVEC CURIOSITÉ
          <span>02 / 03</span>
        </div>

        <h2>
          Plus qu’une traduction.
          <br />
          <span>Une expérience.</span>
        </h2>
      </div>

      <div className="feature-card">
        <div className="feature-icon mint">
          <WandSparkles size={19} />
        </div>

        <span className="feature-number">01</span>

        <h3>Un modèle qui apprend</h3>

        <p>
          Un projet NLP explorant les réseaux récurrents et la
          traduction séquence-à-séquence.
        </p>
      </div>

      <div className="feature-card">
        <div className="feature-icon violet">
          <Languages size={19} />
        </div>

        <span className="feature-number">02</span>

        <h3>Une langue vivante</h3>

        <p>
          La darija marocaine, une langue du quotidien riche en
          nuances et en expressions.
        </p>
      </div>

      <div className="feature-card">
        <div className="feature-icon amber">
          <Mic2 size={19} />
        </div>

        <span className="feature-number">03</span>

        <h3>Un projet expérimental</h3>

        <p>
          Une interface dédiée à un projet académique de Deep
          Learning et de traitement du langage.
        </p>
      </div>
    </section>
  )
}
