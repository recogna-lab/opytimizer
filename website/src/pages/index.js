import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

const FeatureList = [
  {
    title: 'Nature-Inspired Heuristics',
    Svg: require('@site/static/img/feature_dna.svg').default,
    description: (
      <>
        Access dozens of optimization algorithms based on swarms, genetic 
        evolution, and physical phenomena, ready to be used out-of-the-box.
      </>
    ),
  },
  {
    title: 'Extensible and Modular',
    Svg: require('@site/static/img/feature_layers.svg').default,
    description: (
      <>
        Built in Python with a clean architecture. Easily create and 
        experiment with your own algorithms by extending our base classes.
      </>
    ),
  },
  {
    title: 'GPU & Multi-Objective Support',
    Svg: require('@site/static/img/feature_cpu.svg').default,
    description: (
      <>
        Designed for scalability. Optimize complex functions quickly using 
        parallel processing and advanced multi-objective algorithms.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4', 'feature-card')}>
      <div className="text--center">
        <div className="feature-icon-placeholder">
            <Svg />
        </div>
      </div>
      <div className="text--center padding-horiz--md mt-4">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', 'hero-banner')}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title || 'Opytimizer'}</h1>
        <p className="hero__subtitle">
          {siteConfig.tagline || 'The ultimate Python framework for optimization algorithms.'}
        </p>
        <div className="hero-buttons">
          <Link
            className="button button--secondary button--lg"
            to="/docs">
            Get Started ⏱️
          </Link>
          <Link
            className="button button--outline button--lg btn-github"
            href="https://github.com/recogna-lab/opytimizer">
            GitHub
          </Link>
        </div>
        <div className="terminal-window">
          <code>pip install opytimizer</code>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  return (
    <Layout
      title={`Welcome to Opytimizer`}
      description="Python framework for building and experimenting with optimization algorithms.">
      <HomepageHeader />
      <main>
        <section className="features-section">
          <div className="container">
            <div className="row">
              {FeatureList.map((props, idx) => (
                <Feature key={idx} {...props} />
              ))}
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}