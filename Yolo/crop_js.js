body {
    font-family: 'Inter', sans-serif;
    color: #666666;
    background-color: #f7fafc;
    font-size: 16px;
    padding-bottom: 5rem;
}

.headline {
    font-size: 1.25rem;
    font-weight: 600;
}

.text--small {
    font-size: 0.875rem;
}

a {
    color: #606FC7;
    font-weight: 600;
}

a:hover {
    color: #434190;
}

i {
    position: absolute;
    padding: 0.75rem;
    color: #606FC7;
}

span.icon {
    position: absolute;
    padding: 0.5rem;
    right: 0;
}

.header {
    background-color: white;
    padding: 1rem;
    padding-bottom: 2.5rem;
}

.header__grid {
    display: grid;
    grid-template-columns: repeat(1, minmax(0, 1fr));
    grid-template-rows: repeat(4, minmax(0, 1fr));
    gap: 1.5rem;
    max-width: 56rem;
}

.header__logo {
    height: 4rem;
}

.header__label {
    text-transform: uppercase;
    display: block;
    margin-bottom: 0.5rem;
    font-size: 0.875rem;
    font-weight: 600;
    color: #718096;
}

.content {
    padding: 1rem;
    width: 100%;
}

.content__grid {
    display: grid;
    grid-template-columns: repeat(12, minmax(0, 1fr));
    grid-template-rows: repeat(3, minmax(0, 1fr));
    max-width: 56rem;
    column-gap: 1rem;
    row-gap: 2.5rem;
    padding-top: 1rem;
    padding-bottom: 1rem;
}

#imageOptions.content__grid {
    grid-template-rows: repeat(2, minmax(0, 0.5fr));
}

.flex {
    display: flex;
}

.flex-1 {
    flex: 1 1 0%;
}

.relative {
    position: relative;
}

.col-6-m3 {
    grid-column: span 6 / span 6;
}

.col-12-s6-m4, .col-12-m6, .col-12-m8, .col-12 {
    grid-column: span 12 / span 12;
}

.result {
    max-width: 56rem;
}

.result__header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 1rem;
}

.divider {
    border-width: 1px;
    border-color: #cbd5e0;
    margin-top: 2.5rem;
    margin-bottom: 2.5rem;
    height: 0;
}

input:disabled {
    background-color: white;
}

.input {
    border-width: 1px;
    border-color: #cbd5e0;
    border-radius: 0.25rem;
    height: 2.5rem;
    width: 100%;
    padding-left: 0.5rem;
    padding-right: 0.5rem;
}

.input--left {
    border-top-right-radius: 0;
    border-bottom-right-radius: 0;
    margin-right: -1rem;
}

.input__icon {
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}

.input__label {
    margin-bottom: 0.5rem;
    display: block;
}

.bttn {
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
    padding-left: 0.75rem;
    padding-right: 0.75rem;
    background-color: white;
    border-width: 1px;
    border-color: #cbd5e0;
    margin-right: -0.5rem;
    height: 2.5rem;
}

.bttn.fill {
    width: 50%;
}

.bttn:focus {
    outline: 1px dotted;
}

.bttn:hover {
    background-color: #edf2f7;
}

.left {
    border-top-left-radius: 0.25rem;
    border-bottom-left-radius: 0.25rem
