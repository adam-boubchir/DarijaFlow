import { useState } from 'react'
import { BrowserRouter, useLocation, useNavigate } from 'react-router-dom'
import {
  ArrowLeftRight,
  ArrowUpRight,
  Check,
  ChevronDown,
  Copy,
  Languages,
  LoaderCircle,
  Moon,
  Sparkles,
  Zap,
  WandSparkles,
} from 'lucide-react'

import './App.css'
import TrainingDashboard from './TrainingDashboard'
import HowItWorks from './HowItWorks'
import AboutProject from './AboutProject'

const examples = [
  {
    english: 'How are you doing today?',
    label: 'Everyday conversation',
  },
  {
    english: 'I would like a glass of water, please.',
    label: 'Useful phrases',
  },
  {
    english: 'I am learning something new every day.',
    label: 'Learning & growth',
  },
]

type ModelKey = 'vanilla' | 'cifg' | 'wmc'

type Translations = Record<ModelKey, string>

const modelLabels: Record<ModelKey, string> = {
  vanilla: 'Vanilla LSTM',
  cifg: 'CIFG LSTM',
  wmc: 'WMC LSTM',
}

function AppContent() {
  const location = useLocation()
  const navigate = useNavigate()

  const activeView =
    location.pathname === '/training' ? 'training' : 'translator'

  const [sourceText, setSourceText] = useState('')
  const [translations, setTranslations] =
    useState<Translations | null>(null)

  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [copiedModel, setCopiedModel] = useState<ModelKey | null>(null)

  // Appelle le backend pour obtenir les résultats des trois modèles.
  const translateText = async () => {
    const text = sourceText.trim()

    if (!text || isLoading) return

    setIsLoading(true)
    setError('')
    setTranslations(null)
    setCopiedModel(null)

    try {
      const response = await fetch(
        'http://127.0.0.1:8000/translate',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text }),
        },
      )

      if (!response.ok) {
        const errorData = await response.json().catch(() => null)

        throw new Error(
          errorData?.detail ??
            `Le serveur a répondu avec le statut ${response.status}.`,
        )
      }

      const data = await response.json()
      const results = data.translations

      if (
        !results ||
        typeof results.vanilla !== 'string' ||
        typeof results.cifg !== 'string' ||
        typeof results.wmc !== 'string'
      ) {
        throw new Error(
          'Le backend ne renvoie pas les traductions des trois modèles.',
        )
      }

      setTranslations({
        vanilla: results.vanilla,
        cifg: results.cifg,
        wmc: results.wmc,
      })
    } catch (err) {
      setError(
        err instanceof Error
          ? `${err.message} Vérifie que le backend est démarré.`
          : 'Impossible de contacter le service de traduction.',
      )
    } finally {
      setIsLoading(false)
    }
  }

  // Copie la traduction du modèle choisi.
  const copyTranslation = async (model: ModelKey) => {
    const value = translations?.[model]

    if (!value) return

    try {
      await navigator.clipboard.writeText(value)

      setCopiedModel(model)

      window.setTimeout(() => {
        setCopiedModel((current) =>
          current === model ? null : current,
        )
      }, 1800)
    } catch {
      setError(
        'La copie a échoué. Sélectionne le texte puis copie-le manuellement.',
      )
    }
  }

  // Remplit le champ avec une phrase d'exemple.
  const useExample = (text: string) => {
    setSourceText(text)
    setTranslations(null)
    setCopiedModel(null)
    setError('')
  }

  const openTranslator = () => {
    setError('')
    navigate('/')
  }

  const openTrainingResults = () => {
    setError('')
    navigate('/training')
  }

  return (
    <div className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <div className="noise-layer" />

      {/* Navigation principale */}
      <header className="topbar">
        <button
          className="brand"
          type="button"
          onClick={openTranslator}
          aria-label="DarijaFlow, accueil"
        >
          <span className="brand-mark">
            <Languages size={21} strokeWidth={2.2} />
          </span>

          <span className="brand-name">
            darija<span>flow</span>
            <i>.</i>
          </span>
        </button>

        <nav
          className="nav-links"
          aria-label="Navigation principale"
        >
          <button
            className={location.pathname === '/' ? 'nav-active' : ''}
            type="button"
            onClick={openTranslator}
            aria-current={
              location.pathname === '/' ? 'page' : undefined
            }
          >
            Traducteur
          </button>

          <button
            className={
              activeView === 'training' ? 'nav-active' : ''
            }
            type="button"
            onClick={openTrainingResults}
            aria-current={
              activeView === 'training' ? 'page' : undefined
            }
          >
            Training Results
          </button>

          <button
            className={
              location.pathname === '/how-it-works'
                ? 'nav-active'
                : ''
            }
            type="button"
            onClick={() => navigate('/how-it-works')}
            aria-current={
              location.pathname === '/how-it-works'
                ? 'page'
                : undefined
            }
          >
            Comment ça marche
          </button>

          <button
            className={
              location.pathname === '/about' ? 'nav-active' : ''
            }
            type="button"
            onClick={() => navigate('/about')}
            aria-current={
              location.pathname === '/about' ? 'page' : undefined
            }
          >
            Le projet
          </button>
        </nav>

        <div className="topbar-right">
          <span className="status-pill">
            <span className="status-dot" />
            Projet NLP
          </span>

          <button
            className="icon-button theme-button"
            type="button"
            aria-label="Thème sombre"
            title="Thème sombre"
          >
            <Moon size={17} />
          </button>
        </div>
      </header>

      <main id="top">
        {location.pathname === '/training' ? (
          <TrainingDashboard />
        ) : location.pathname === '/how-it-works' ? (
          <HowItWorks />
        ) : location.pathname === '/about' ? (
          <AboutProject />
        ) : (
          <>
            {/* Section d'accueil */}
            <section className="hero-section">
              <div className="hero-copy">
                <div className="eyebrow">
                  <span className="eyebrow-icon">
                    <Sparkles size={14} />
                  </span>

                  LA LANGUE, SANS FRONTIÈRES

                  <span className="eyebrow-line" />
                </div>

                <h1>
                  English in.
                  <br />
                  <span className="gradient-text">
                    Darija out.
                  </span>
                </h1>

                <p className="hero-description">
                  Les mots voyagent. Découvrez une nouvelle façon de
                  passer de l’anglais à la darija marocaine, avec un
                  projet de traduction basé sur le Deep Learning.
                </p>

                <div className="hero-meta">
                  <div
                    className="avatar-stack"
                    aria-hidden="true"
                  >
                    <span>EN</span>
                    <span>MA</span>
                    <span>AI</span>
                  </div>

                  <span>
                    <strong>Anglais</strong>
                    <span className="meta-arrow">↔</span>
                    <strong>Darija marocaine</strong>
                  </span>
                </div>
              </div>

              <div
                className="hero-visual"
                aria-hidden="true"
              >
                <div className="orbit orbit-large" />
                <div className="orbit orbit-small" />
                <div className="visual-glow" />

                <div className="visual-core">
                  <span className="core-letter">د</span>

                  <span className="core-spark">
                    <Sparkles size={20} />
                  </span>
                </div>

                <div className="floating-chip chip-english">
                  <span className="chip-dot" />
                  Hello there
                </div>

                <div
                  className="floating-chip chip-darija"
                  dir="rtl"
                >
                  سلام، لاباس؟

                  <span className="chip-check">
                    <Check size={12} />
                  </span>
                </div>

                <div className="visual-caption">
                  <span className="caption-pulse" />
                  Neural translation
                  <span>·</span>
                  Language model
                </div>
              </div>
            </section>

            {/* Espace de traduction */}
            <section
              className="translator-section"
              id="translator"
            >
              <div className="section-heading">
                <div>
                  <div className="section-kicker">
                    L’ESPACE DE TRADUCTION
                    <span>01 / 03</span>
                  </div>

                  <h2>
                    Une idée. Deux langues. <span>Un pont.</span>
                  </h2>
                </div>

                <div className="secure-note">
                  <span className="secure-icon">
                    <Zap size={14} />
                  </span>

                  Pensé pour le naturel
                </div>
              </div>

              <div className="translator-card">
                <div className="translator-topline">
                  <div className="translator-mode">
                    <span className="mode-indicator" />
                    MODE TRADUCTION
                  </div>

                  <div className="model-tag">
                    <span className="model-spark">
                      <WandSparkles size={13} />
                    </span>

                    English → Darija

                    <ChevronDown size={13} />
                  </div>
                </div>

                <div className="translation-grid">
                  {/* Champ de texte anglais */}
                  <div className="language-panel source-panel">
                    <div className="language-header">
                      <div className="language-label">
                        <span className="language-symbol">
                          EN
                        </span>

                        <span>
                          <strong>English</strong>
                          <small>Langue source</small>
                        </span>
                      </div>

                      <span className="panel-index">01</span>
                    </div>

                    <textarea
                      value={sourceText}
                      onChange={(event) => {
                        setSourceText(event.target.value)
                      }}
                      onKeyDown={(event) => {
                        if (
                          (event.ctrlKey || event.metaKey) &&
                          event.key === 'Enter'
                        ) {
                          void translateText()
                        }
                      }}
                      placeholder="Type something wonderful..."
                      maxLength={1000}
                      aria-label="Texte anglais à traduire"
                    />

                    <div className="panel-footer">
                      <span>
                        {sourceText.length} / 1000 caractères
                      </span>

                      <button
                        className="small-action"
                        type="button"
                        onClick={() => {
                          setSourceText('')
                          setTranslations(null)
                          setCopiedModel(null)
                          setError('')
                        }}
                        disabled={!sourceText}
                      >
                        Effacer
                      </button>
                    </div>
                  </div>

                  {/* Icône entre les deux langues */}
                  <div
                    className="swap-column"
                    aria-hidden="true"
                  >
                    <div className="swap-orb">
                      <ArrowLeftRight size={17} />
                    </div>
                  </div>

                  {/* Résultats des trois modèles */}
                  <div className="language-panel target-panel models-panel">
                    <div className="language-header">
                      <div className="language-label">
                        <span className="language-symbol darija-symbol">
                          MA
                        </span>

                        <span>
                          <strong>
                            Traductions en darija
                          </strong>
                          <small>
                            Comparaison des trois modèles LSTM
                          </small>
                        </span>
                      </div>

                      <span className="panel-index">03</span>
                    </div>

                    <div
                      className="model-results"
                      aria-live="polite"
                    >
                      {isLoading ? (
                        <div className="loading-state">
                          <LoaderCircle
                            className="spinner"
                            size={22}
                          />

                          <span>
                            Les trois modèles génèrent leurs
                            traductions...
                          </span>
                        </div>
                      ) : translations ? (
                        (
                          [
                            ['vanilla', modelLabels.vanilla],
                            ['cifg', modelLabels.cifg],
                            ['wmc', modelLabels.wmc],
                          ] as const
                        ).map(([key, label]) => (
                          <article
                            className="model-result-card"
                            key={key}
                          >
                            <div className="model-result-heading">
                              <span
                                className={`model-badge model-${key}`}
                              >
                                {label}
                              </span>

                              <button
                                className="small-action copy-action"
                                type="button"
                                onClick={() =>
                                  void copyTranslation(key)
                                }
                                disabled={!translations[key]}
                                aria-label={`Copier le résultat ${label}`}
                              >
                                {copiedModel === key ? (
                                  <Check size={13} />
                                ) : (
                                  <Copy size={13} />
                                )}

                                {copiedModel === key
                                  ? 'Copié'
                                  : 'Copier'}
                              </button>
                            </div>

                            <p
                              className="model-result-text"
                              dir="auto"
                            >
                              {translations[key] ||
                                'Aucune traduction générée.'}
                            </p>
                          </article>
                        ))
                      ) : (
                        <div className="empty-state">
                          <span className="empty-icon">
                            <Languages size={22} />
                          </span>

                          <p>
                            Vos trois traductions apparaîtront ici.
                          </p>

                          <small>
                            Une phrase anglaise, trois modèles,
                            trois résultats.
                          </small>
                        </div>
                      )}
                    </div>

                    <div className="panel-footer">
                      <span>
                        {translations
                          ? '3 modèles exécutés'
                          : 'En attente du texte'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Message d'erreur */}
                {error && (
                  <div
                    className="error-message"
                    role="alert"
                  >
                    {error}
                  </div>
                )}

                {/* Bouton de traduction */}
                <div className="translator-actions">
                  <div className="hint">
                    <span className="hint-key">↵</span>

                    <span>
                      <strong>Ctrl + Entrée</strong> pour traduire
                    </span>
                  </div>

                  <button
                    className="translate-button"
                    onClick={() => void translateText()}
                    disabled={!sourceText.trim() || isLoading}
                    type="button"
                  >
                    {isLoading ? (
                      <LoaderCircle
                        className="spinner"
                        size={17}
                      />
                    ) : (
                      <Sparkles size={17} />
                    )}

                    <span>
                      {isLoading
                        ? 'Traduction en cours'
                        : 'Traduire en darija'}
                    </span>

                    {!isLoading && (
                      <ArrowUpRight size={17} />
                    )}
                  </button>
                </div>
              </div>

              {/* Exemples de phrases */}
              <div className="examples-row">
                <div className="examples-label">
                  <span className="examples-spark">
                    <Sparkles size={13} />
                  </span>

                  <span>Besoin d’inspiration ?</span>
                </div>

                {examples.map((example) => (
                  <button
                    className="example-chip"
                    type="button"
                    key={example.english}
                    title={example.label}
                    onClick={() =>
                      useExample(example.english)
                    }
                  >
                    <span>{example.english}</span>
                    <ArrowUpRight size={13} />
                  </button>
                ))}
              </div>
            </section>
          </>
        )}
      </main>

      {/* Pied de page */}
      <footer className="footer">
        <button
          className="footer-brand"
          type="button"
          onClick={openTranslator}
          aria-label="Retour à l'accueil"
        >
          darija<span>flow</span>
          <i>.</i>
        </button>

        <p>
          Fait avec curiosité, du code et beaucoup de thé à la menthe.
        </p>

        <span className="footer-tag">
          <span className="status-dot" />
          Projet NLP · 2026
        </span>
      </footer>
    </div>
  )
}

function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  )
}

export default App